import numpy as np
import torch
import torch.nn.functional as F
from torch_struct import SemiMarkovCRF
import argparse
from sklearn.mixture import GaussianMixture
from data import labels_to_spans, spans_to_labels, rle_spans

class FeedForward(torch.nn.Module):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--n_layers', type=int, default=1)
        parser.add_argument('--hidden_size', type=int, default=100)

    def __init__(self, args, input_size, output_size):
        super(FeedForward, self).__init__()
        self.args = args
        self.in_layer = torch.nn.Linear(input_size, args.hidden_size, bias=True)
        self.out_layer = torch.nn.Linear(args.hidden_size, output_size, bias=True)
        for i in range(args.n_layers):
            name = 'cell{}'.format(i)
            cell = torch.nn.Linear(args.hidden_size, args.hidden_size, bias=True)
            setattr(self, name, cell)

    def reset_parameters(self):
        self.in_layer.reset_parameters()
        self.out_layer.reset_parameters()
        for i in range(self.args.n_layers):
            name = 'cell{}'.format(i)
            getattr(self, name).reset_parameters()

    def init_identity(self):
        self.in_layer.weight.data.zero_()
        self.in_layer.bias.data.zero_()
        self.out_layer.weight.data.zero_()
        self.out_layer.bias.data.zero_()
        for i in range(self.args.n_layers):
            name = 'cell{}'.format(i)
            getattr(self, name).weight.data.zero_()
            getattr(self, name).bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.in_layer(x))
        for i in range(self.args.n_layers):
            name = 'cell{}'.format(i)
            x = F.relu(getattr(self, name)(x))
        return self.out_layer(x)

class NICE(torch.nn.Module):
    @classmethod
    def add_args(cls, parser):
        FeedForward.add_args(parser)
        parser.add_argument('--n_blocks', type=int, default=4)
        parser.add_argument('--scale', action='store_true')
        parser.add_argument('--scale_no_zero', action='store_true')

    def __init__(self, args, input_size):
        super(NICE, self).__init__()
        self.args = args
        for i in range(args.n_blocks):
            name = 'block{}'.format(i)
            block = FeedForward(args, input_size // 2, input_size // 2)
            setattr(self, name, block)
            if args.scale:
                scale_name = 'scale_block{}'.format(i)
                scale_block = FeedForward(args, input_size // 2, input_size // 2)
                if not args.scale_no_zero:
                    scale_block.init_identity()
                setattr(self, scale_name, scale_block)

    def reset_parameters(self):
        for i in range(self.args.n_blocks):
            name = 'block{}'.format(i)
            getattr(self, name).reset_parameters()
            if self.args.scale:
                name = 'scale_block{}'.format(i)
                getattr(self, name).reset_parameters()

    def forward(self, x):
        jacobian_loss = torch.zeros(x.size(0), device=x.device, requires_grad=False)

        n_feat = x.size()[-1]
        for i in range(self.args.n_blocks):
            name = 'cell{}'.format(i)
            h1, h2 = torch.split(x, n_feat // 2, dim=-1)
            if i % 2 == 1:
                h1, h2 = h2, h1
            t = getattr(self, name)(h1)
            if self.args.scale:
                s = getattr(self, 'scale_block{}'.format(i))(h1)
                jacobian_loss += s.sum(dim=-1).sum(dim=-1)
                h2_p = torch.exp(s) * h2 + t
            else:
                h2_p = h2 + t
            if i % 2 == 1:
                h1, h2_p = h2_p, h1
            h = torch.cat((h1, h2_p), dim=-1)
        return h, jacobian_loss

class ResidualLayer(torch.nn.Module):
    def __init__(self, dim=100):
        super(ResidualLayer, self).__init__()
        self.h1 = torch.nn.Linear(dim, dim)
        self.h2 = torch.nn.Linear(dim, dim)

    def forward(self, x):
        z = F.relu(self.h1(x))
        z = F.relu(self.h2(z))
        return x + z

class SemiMarkovModule(torch.nn.Module):
    @classmethod
    def add_args(cls, parser):
        NICE.add_args(parser)
        parser.add_argument('--sm_max_span_length', type=int, default=20)
        parser.add_argument('--sm_supervised_state_smoothing', type=float, default=1e-2)
        parser.add_argument('--sm_supervised_length_smoothing', type=float, default=1e-1)
        parser.add_argument('--sm_supervised_method', choices=['closed_form', 'gradient_based', 'closed-then-gradient'], default='closed_form')
        parser.add_argument('--sm_feature_projection', action='store_true')
        parser.add_argument('--sm_init_non_projection_parameters_from')
        parser.add_argument('--lr', type=float, default=1e-1)

    def __init__(self, args, n_classes, n_dims, allow_self_transitions=True):
        super(SemiMarkovModule, self).__init__()
        self.args = args
        self.n_classes = n_classes
        self.input_feature_dim = n_dims
        self.feature_dim = n_dims
        self.allow_self_transitions = allow_self_transitions
        self.max_k = args.sm_max_span_length
        self.learning_rate = args.lr
        self.init_params()
        self.init_projector()

    def init_params(self):
        """Create torch differentiable params"""
        poisson_log_rates = torch.zeros(self.n_classes, dtype=torch.float)
        self.poisson_log_rates = torch.nn.Parameter(poisson_log_rates, requires_grad=True)

        gaussian_means = torch.zeros(self.n_classes, self.feature_dim, dtype=torch.float)
        self.gaussian_means = torch.nn.Parameter(gaussian_means, requires_grad=True)

        gaussian_cov = torch.eye(self.feature_dim, dtype=torch.float)
        self.gaussian_cov = torch.nn.Parameter(gaussian_cov, requires_grad=False)

        transition_logits = torch.zeros(self.n_classes, self.n_classes, dtype=torch.float)
        self.transition_logits = torch.nn.Parameter(transition_logits, requires_grad=True)

        init_logits = torch.zeros(self.n_classes, dtype=torch.float)
        self.init_logits = torch.nn.Parameter(init_logits, requires_grad=True)
        torch.nn.init.uniform_(self.init_logits, 0, 1)

    def init_projector(self):
        if self.args.sm_feature_projection:
            self.feature_projector = NICE(self.args, input_size=self.feature_dim)
        else:
            self.feature_projector = None

    def initialize_gaussian(self, data, lengths):
        b, _, d = data.size()
        assert lengths.size(0) == b
        feats = []
        for i in range(b):
            feats.append(data[i, :lengths[i]])
        feats = torch.cat(feats, dim=0)
        if self.feature_projector:
            feats, _ = self.feature_projector(feats)
        assert d == self.feature_dim
        mean = feats.mean(dim=0, keepdim=True)
        self.gaussian_means.data.zero_()
        self.gaussian_means.data.add_(mean.expand((self.n_classes, self.feature_dim)))
        self.gaussian_cov.data = torch.diag(feats.var(dim=0))

    def initialize_supervised(self, feature_list, label_list, overrides=['mean', 'cov', 'init', 'trans', 'lengths']):
        emission_gmm, stats = semimarkov_sufficient_stats(feature_list, label_list, covariance_type='tied_diag', n_classes=self.n_classes, max_k=self.max_k)
        if 'init' in overrides:
            init_probs = (stats['span_start_counts'] + self.args.sm_supervised_state_smoothing) /\
                float(stats['instance_count'] + self.args.sm_supervised_state_smoothing * self.n_classes)
            init_probs[np.isnan(init_probs)] = 0
            self.init_logits.data.zero_()
            self.init_logits.data.add_(torch.from_numpy(init_probs).to(device=self.init_logits.device).log())
            self.init_logits.requires_grad = False
        if 'trans' in overrides:
            smoothed_trans_counts = stats['span_transition_counts'] + self.args.sm_supervised_state_smoothing
            trans_probs = smoothed_trans_counts / smoothed_trans_counts.sum(axis=0)[None, :]
            trans_probs[np.isnan(trans_probs)] = 0
            self.transition_logits.data.zero_()
            self.transition_logits.data.add_(torch.from_numpy(trans_probs).to(device=self.transition_logits.device).log())
            self.transition_logits.requires_grad = False
        if 'lengths' in overrides:
            mean_lengths = (stats['span_lengths'] + self.args.sm_supervised_length_smoothing) /\
                (stats['span_counts'] + self.args.sm_supervised_length_smoothing)
            self.poisson_log_rates.data.zero_()
            self.poisson_log_rates.data.add_(torch.from_numpy(mean_lengths).to(device=self.poisson_log_rates.device).log())
            self.poisson_log_rates.requires_grad = False
        if 'mean' in overrides:
            self.gaussian_means.data.zero_()
            self.gaussian_means.data.add_(torch.from_numpy(emission_gmm.means_).to(device=self.gaussian_means.device, dtype=torch.float))
            self.gaussian_means.requires_grad = False
        if 'cov' in overrides:
            self.gaussian_cov.data.zero_()
            self.gaussian_cov.data.add_(torch.diag(torch.from_numpy(emission_gmm.covariances_[0]).to(device=self.gaussian_cov.device, dtype=torch.float)))

    def fit_supervised(self, feature_list, label_list):
        if self.feature_projector is not None:
            raise NotImplementedError('fit_supervised with feature projector')

        emission_gmm, stats = semimarkov_sufficient_stats(feature_list, label_list, covariance_type='tied_diag', n_classes=self.n_classes, max_k=self.max_k)

        init_probs = (stats['span_start_counts'] + self.args.sm_supervised_state_smoothing) /\
            float(stats['instance_count'] + self.args.sm_supervised_state_smoothing * self.n_classes)
        init_probs[np.isnan(init_probs)] = 0
        self.init_logits.data.zero_()
        self.init_logits.data.add_(torch.from_numpy(init_probs).to(device=self.init_logits.device).log())

        smoothed_trans_counts = stats['span_transition_counts'] + self.args.sm_supervised_state_smoothing
        trans_probs = smoothed_trans_counts / smoothed_trans_counts.sum(axis=0)[None, :]
        trans_probs[np.isnan(trans_probs)] = 0
        self.transition_logits.data.zero_()
        self.transition_logits.data.add_(torch.from_numpy(trans_probs).to(device=self.transition_logits.device).log())

        mean_lengths = (stats['span_lengths'] + self.args.sm_supervised_length_smoothing) /\
            (stats['span_counts'] + self.args.sm_supervised_length_smoothing)
        self.poisson_log_rates.data.zero_()
        self.poisson_log_rates.data.add_(torch.from_numpy(mean_lengths).to(device=self.poisson_log_rates.device).log())

        self.gaussian_means.data.zero_()
        self.gaussian_means.data.add_(torch.from_numpy(emission_gmm.means_).to(device=self.gaussian_means.device, dtype=torch.float))

        self.gaussian_cov.data.zero_()
        self.gaussian_cov.data.add_(torch.diag(torch.from_numpy(emission_gmm.covariances_[0]).to(device=self.gaussian_cov.device, dtype=torch.float)))

    def transition_log_probs(self, valid_classes):
        """Mask out invalid classes and apply softmax to transition logits"""
        transition_logits = self.transition_logits

        if valid_classes is not None:
            transition_logits = transition_logits[valid_classes][:, valid_classes]
            n_classes = len(valid_classes)
        else:
            n_classes = self.n_classes

        if self.allow_self_transitions:
            masked = transition_logits
        else:
            masked = transition_logits.masked_fill(torch.eye(n_classes, device=self.transition_logits.device, dtype=bool), -1e9)

        return F.log_softmax(masked, dim=0)

    def emission_log_probs(self, features, valid_classes):
        """Compute likelihood of emissions for each class"""
        if valid_classes is None:
            class_indices = torch.LongTensor(list(range(self.n_classes))).to(self.gaussian_means.device)
        else:
            class_indices = valid_classes
        class_means = self.gaussian_means[class_indices]

        B, _, d = features.size()
        if class_means.dim() == 2:
            num_classes, d_ = class_means.size()
            assert d == d_, (d, d_)
            class_means = class_means.unsqueeze(0)
        else:
            _, num_classes, d_ = class_means.size()
            assert d == d_, (d, d_)
        class_means = class_means.expand(B, num_classes, d)
        scale_tril = self.gaussian_cov.sqrt()

        log_probs = []
        for c in range(num_classes):
            means_ = class_means[:, c, :]
            dist = torch.distributions.MultivariateNormal(loc=means_, scale_tril=scale_tril)
            log_probs.append(dist.log_prob(features.transpose(0, 1)).transpose(0, 1).unsqueeze(-1))
        return torch.cat(log_probs, dim=2)

    def initial_log_probs(self, valid_classes):
        """Mask out invalid classes and apply softmax to initial logits"""
        logits = self.init_logits

        if valid_classes is not None:
            logits = logits[valid_classes]

        return F.log_softmax(logits, dim=0)

    def length_log_probs(self, valid_classes):
        """Compute likelihood of lengths for each class from poisson"""
        if valid_classes is None:
            class_indices = torch.LongTensor(list(range(self.n_classes))).to(self.poisson_log_rates.device)
            n_classes = self.n_classes
        else:
            class_indices = valid_classes
            n_classes = len(class_indices)
        log_rates = self.poisson_log_rates[class_indices]

        time_steps = torch.arange(self.max_k, device=log_rates.device, dtype=torch.float).unsqueeze(-1).expand(self.max_k, n_classes)
        if self.max_k == 1:
            return torch.FloatTensor([0, -1000]).unsqueeze(-1).expand(2, n_classes).to(log_rates.device)
        poissons = torch.distributions.Poisson(torch.exp(log_rates))
        if log_rates.dim() == 2:
            time_steps = time_steps.unsqueeze(1).expand(self.max_k, log_rates.size(0), n_classes)
            return poissons.log_prob(time_steps).transpose(0, 1)
        else:
            assert log_rates.dim() == 1
            return poissons.log_prob(time_steps)

    def log_hsmm(self, transition_scores, emission_scores, init_scores, length_scores, lengths, add_eos, all_batched=False):
        b, N_1, C_1 = emission_scores.shape
        if all_batched:
            _, K, C_ = length_scores.shape
            assert C_1 == C_
        else:
            K, C_ = length_scores.shape
            assert C_1 == C_
            transition = transition_scores.unsqueeze(0).expand(b, C_1, C_1)
            length_scores = length_scores.unsqueeze(0).expand(b, K, C_1)
            init = init_scores.unsqueeze(0).expand(b, C_1)

        if K > N_1:
            K = N_1
            length_scores = length_scores[:, :K]

        if add_eos:
            N = N_1 + 1
            C = C_1 + 1

            transition_augmented = torch.full((b, C, C), -1e9, device=transition.device)
            transition_augmented[:, :C_1, :C_1] = transition
            transition_augmented[:, C_1, :] = 0

            init_augmented = torch.full((b, C), -1e9, device=init.device)
            init_augmented[:, :C_1] = init

            length_scores_augmented = torch.full((b, K, C), -1e9, device=length_scores.device)
            length_scores_augmented[:, :, :C_1] = length_scores
            if length_scores_augmented.size(1) > 1:
                length_scores_augmented[:, 1, C_1] = 0
            else:
                length_scores_augmented[:, 0, C_1] = 0

            emission_augmented = torch.full((b, N, C), -1e9, device=emission_scores.device)
            for i, length in enumerate(lengths):
                assert emission_augmented[i, :length, :C_1].size() == emission_scores[i, :length].size()
                emission_augmented[i, :length, :C_1] = emission_scores[i, :length]
                emission_augmented[i, length, C_1] = 0

            lengths_augmented = lengths + 1
        else:
            N = N_1
            C = C_1

            transition_augmented = transition
            init_augmented = init
            length_scores_augmented = length_scores
            emission_augmented = emission_scores
            lengths_augmented = lengths

        scores = torch.zeros(b, N - 1, K, C, C, device=emission_scores.device).type_as(emission_scores)
        scores[:, :, :, :, :] += transition_augmented.view(b, 1, 1, C, C)
        scores[:, 0, :, :, :] += init_augmented.view(b, 1, 1, C)
        scores[:, :, :, :, :] += length_scores_augmented.view(b, 1, K, 1, C)

        for k in range(1, K):
            summed = sliding_sum(emission_augmented, k).view(b, N, 1, C)
            for i in range(b):
                length = lengths_augmented[i]
                scores[i, :length - 1, k, :, :] += summed[i, :length - 1]
                scores[i, length - 1 - k, k, :, :] += emission_augmented[i, length - 1].view(C, 1)
        return scores

    def score_features(self, features, lengths, valid_classes, add_eos):
        self.kl = torch.autograd.Variable(torch.zeros(features.size(0)).to(features.device), requires_grad=True)

        if self.feature_projector is not None:
            projected_features, log_det = self.feature_projector(features)
        else:
            projected_features = features
            log_det = torch.zeros(features.size(0), device=features.device, requires_grad=False)

        scores = self.log_hsmm(
            self.transition_log_probs(valid_classes),
            self.emission_log_probs(projected_features, valid_classes),
            self.initial_log_probs(valid_classes),
            self.length_log_probs(valid_classes),
            lengths,
            add_eos=add_eos
        )

        return scores, log_det

    def viterbi(self, features, lengths, valid_classes_per_instance, add_eos=True):
        if valid_classes_per_instance is not None:
            valid_classes = valid_classes_per_instance[0]
            C = len(valid_classes)
        else:
            valid_classes = None
            C = self.n_classes

        scores, _ = self.score_features(features, lengths, valid_classes, add_eos=add_eos)

        if add_eos:
            eos_lengths = lengths + 1
        else:
            eos_lengths = lengths

        dist = SemiMarkovCRF(scores, lengths=eos_lengths)
        pred_spans, _ = dist.struct.from_parts(dist.argmax)

        pred_spans_unmap = pred_spans.detach().cpu()
        if valid_classes is not None:
            mapping = {index: cls.item() for index, cls in enumerate(valid_classes)}
            assert len(mapping.values()) == len(mapping), 'valid classes must be unique'
            assert -1 not in mapping.values()
            mapping[-1] = -1
            mapping[C] = self.n_classes
            pred_spans_unmap.apply_(lambda x: mapping[x])

        return pred_spans_unmap.to(features.device)

    def log_likelihood(self, features, lengths, valid_classes_per_instance, add_eos=True):
        if valid_classes_per_instance is not None:
            valid_classes = valid_classes_per_instance[0]
            C = len(valid_classes)
        else:
            valid_classes = None
            C = self.n_classes

        scores, log_det = self.score_features(features, lengths, valid_classes, add_eos=add_eos)

        K = scores.size(2)
        assert K <= self.max_k or (self.max_k == 1 and K == 2)

        if add_eos:
            eos_lengths = lengths + 1
            eos_C = C + 1
        else:
            eos_lengths = lengths
            eoc_C = C

        dist = SemiMarkovCRF(scores, lengths=eos_lengths)
        log_likelihood = dist.partition.mean()

        return log_likelihood, log_det.mean()

    def add_eos(self, spans, lengths):
        b, _ = spans.size()
        augmented = torch.cat([spans, torch.full([b, -1], -1, device=spans.device, dtype=torch.long)], dim=1)
        augmented[torch.arange(b), lengths] = self.n_classes
        return augmented

    def trim(self, spans, lengths):
        b, _ = spans.size()
        seqs = []
        for i in range(b):
            seqs.append(spans[i, :lengths[i]])
        return seqs

def sliding_sum(inputs, k):
    batch_size = inputs.size(0)
    assert k > 0
    if k == 1:
        return inputs
    sliding_windows = F.unfold(inputs.unsqueeze(1), kernel_size=(k, 1), padding=(k, 0)).reshape(batch_size, k, -1, inputs.size(-1))
    sliding_summed = sliding_windows.sum(dim=1)
    ret = sliding_summed[:, k:-1, :]
    assert ret.shape == inputs.shape
    return ret

def semimarkov_sufficient_stats(feature_list, label_list, covariance_type, n_classes, max_k=None):
    assert len(feature_list) == len(label_list)
    if covariance_type == 'tied_diag':
        emissions = GaussianMixture(n_classes, covariance_type='diag')
    else:
        emissions = GaussianMixture(n_classes, covariance_type=covariance_type)
    X_l = []
    r_l = []

    span_counts = np.zeros(n_classes, dtype=np.float32)
    span_lengths = np.zeros(n_classes, dtype=np.float32)
    span_start_counts = np.zeros(n_classes, dtype=np.float32)
    span_transition_counts = np.zeros((n_classes, n_classes), dtype=np.float32)
    instance_count = 0
    for X, labels in zip(feature_list, label_list):
        X = X.cpu(); labels = labels.cpu()
        X_l.append(X)
        r = np.zeros((X.shape[0], n_classes))
        r[np.arange(X.shape[0]), labels] = 1
        assert r.sum() == X.shape[0]
        r_l.append(r)
        spans = labels_to_spans(labels.unsqueeze(0), max_k)
        spans = rle_spans(spans, torch.LongTensor([spans.size(1)]))[0]
        prev = None
        for idx, (symbol, length) in enumerate(spans):
            if idx == 0:
                span_start_counts[symbol] += 1
            span_counts[symbol] += 1
            span_lengths[symbol] += length
            if prev is not None:
                span_transition_counts[symbol, prev] += 1
            prev = symbol
        instance_count += 1

    X_arr = np.vstack(X_l)
    r_arr = np.vstack(r_l)
    emissions._initialize(X_arr, r_arr)
    if covariance_type == 'tied_diag':
        cov, prec_chol = get_diagonal_covariances(X_arr)
        emissions.covariances_[:] = np.copy(cov)
        emissions.precisions_cholesky_[:] = np.copy(prec_chol)
    return emissions, {
        'span_counts': span_counts,
        'span_lengths': span_lengths,
        'span_start_counts': span_start_counts,
        'span_transition_counts': span_transition_counts,
        'instance_count': instance_count
    }

def get_diagonal_covariances(data):
    model = GaussianMixture(n_components=1, covariance_type='diag')
    responsibilities = np.ones((data.shape[0], 1))
    model._initialize(data, responsibilities)
    return model.covariances_, model.precisions_cholesky_

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    SemiMarkovModule.add_args(parser)
    args = parser.parse_args()

    SemiMarkovModule(args, 10, 10)
