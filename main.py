import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from hsmm import SemiMarkovModule, labels_to_spans, spans_to_labels
import argparse

class ToyDataset(Dataset):
    def __init__(self, labels, features, lengths, valid_classes, max_k):
        self.labels = labels
        self.features = features
        self.lengths = lengths
        self.valid_classes = valid_classes
        self.max_k = max_k

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, index):
        labels = self.labels[index]
        spans = labels_to_spans(labels.unsqueeze(0), max_k=self.max_k).squeeze(0)
        return {
            'labels': self.labels[index],
            'features': self.features[index],
            'lengths': self.lengths[index],
            'valid_classes': self.valid_classes[index],
            'spans': spans
        }

def synthetic_data(num_points=200, C=3, N=20, K=5, classes_per_seq=None):
    """Creates synthetic data consisting of semimarkov sequences, featurized as one-hots plus noise

    Parameters
    ----------
    num_points : int
        number of sequences
    C : int
        total number of classes
    N : int
        length of each sequence
    K : int
        max action length (same class repeating)
    classes_per_seq : int
        max number of classes present in each sequence
    """

    def make_features(class_labels, shift_constant=1.0):
        """converts labels to one-hot plus some gaussian noise

        Parameters
        ----------
        class_labels : sequences of labels
            Description of parameter `class_labels`.
        shift_constant : type
            Description of parameter `shift_constant`.
        """
        batch_size_, N_ = class_labels.size()
        f = torch.randn((batch_size_, N_, C))
        shift = torch.zeros_like(f)
        shift.scatter_(2, class_labels.unsqueeze(2), shift_constant)
        return shift + f

    labels = []
    lengths = []
    valid_classes = []
    for i in range(num_points):
        if i == 0:
            length = N
        else:
            length = random.randint(K, N)
        lengths.append(length)
        seq = []
        current_step = 0
        if classes_per_seq is not None:
            assert classes_per_seq <= C
            valid_classes_ = np.random.choice(list(range(C)), size=classes_per_seq, replace=False)
        else:
            valid_classes_ = list(range(C))
        valid_classes.append(valid_classes_)
        while len(seq) < N:
            step_len = random.randint(1, K-1)
            seq_ = valid_classes_[current_step % len(valid_classes_)]
            seq.extend([seq_] * step_len)
            current_step += 1
        seq = seq[:N]
        labels.append(seq)
    labels = torch.LongTensor(labels)
    features = make_features(labels)
    lengths = torch.LongTensor(lengths)
    valid_classes = [torch.LongTensor(c) for c in valid_classes]

    return labels, features, lengths, valid_classes

def predict(model, dataloader):
    items = []
    token_match, token_total = 0, 0
    for batch in dataloader:
        features = batch['features']
        lengths = batch['lengths']
        gold_spans = batch['spans']
        valid_classes = batch['valid_classes']

        batch_size = features.size(0)
        N_ = lengths.max().item()
        features = features[:, :N_, :]
        gold_spans = gold_spans[:, :N_]

        pred_spans = model.viterbi(features, lengths, valid_classes_per_instance=valid_classes, add_eos=True)
        gold_labels = spans_to_labels(gold_spans)
        pred_labels = spans_to_labels(pred_spans)

        gold_labels_trim = model.trim(gold_labels, lengths)
        pred_labels_trim = model.trim(pred_labels, lengths)

        assert len(gold_labels_trim) == batch_size
        assert len(pred_labels_trim) == batch_size

        for i in range(batch_size):
            item = {
                'length': lengths[i].item(),
                'gold_spans': gold_spans[i],
                'pred_spans': pred_spans[i],
                'gold_labels': gold_labels[i],
                'pred_labels': pred_labels[i],
                'gold_labels_trim': gold_labels_trim[i],
                'pred_labels_trim': pred_labels_trim[i]
            }
            items.append(item)
            token_match += (gold_labels_trim[i] == pred_labels_trim[i]).sum().item()
            token_total += pred_labels_trim[i].size(0)
    accuracy = 100. * token_match / token_total
    return accuracy, items

def train_supervised(train_dset):
    C = train_dset[0]['features'].size()[-1]

    parser = argparse.ArgumentParser()
    SemiMarkovModule.add_args(parser)
    args = parser.parse_args()
    model = SemiMarkovModule(args, C, C, allow_self_transitions=True)

    train_features = []
    train_labels = []
    for i in range(len(train_dset)):
        sample = train_dset[i]
        train_features.append(sample['features'])
        train_labels.append(sample['labels'])
    model.fit_supervised(train_features, train_labels)

    return model

def train_unsupervised(train_dset):
    C = train_dset[0]['features'].size()[-1]

    parser = argparse.ArgumentParser()
    SemiMarkovModule.add_args(parser)
    args = parser.parse_args()
    model = SemiMarkovModule(args, C, C, allow_self_transitions=True)

    dataloader = DataLoader(train_dset, batch_size=10)
    s = next(iter(dataloader))
    features = s['features']
    lengths = s['lengths']
    model.initialize_gaussian(features, lengths)

    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.2, verbose=True, patience=1, min_lr=1e-4, threshold=1e-5)

    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        multi_batch_losses = []
        nlls = []
        kls = []
        log_dets = []
        n = 0
        n_frames = 0
        train_nll = 0
        train_kl = 0
        train_log_det = 0
        for batch_idx, batch in enumerate(dataloader):
            features = batch['features']
            lengths = batch['lengths']
            n += len(lengths)
            n_frames += lengths.sum().item()

            ll, log_det = model.log_likelihood(features, lengths)
            #why loss becoming nan?

            nll = -ll
            kl = model.kl.mean()
            loss_ = nll - log_det + kl

            multi_batch_losses.append(loss_)
            nlls.append(nll.item())
            kls.append(kl.item())
            log_dets.append(log_det.item())

            train_nll += (nll.item() * n)
            train_kl += (kl.item() * n)
            train_log_det += (log_det.item() * n)

            losses.append(loss_.item())

            if len(multi_batch_losses) >= 1:
                loss = sum(multi_batch_losses) / len(multi_batch_losses)
                loss.backward()

                multi_batch_losses = []

                if batch_idx % 1 == 0:
                    param_norm = sum([p.norm() ** 2 for p in model.parameters() if p.requires_grad]).item() ** .5
                    gparam_norm = sum([p.grad.norm() ** 2 for p in model.parameters() if p.requires_grad and p.grad is not None]).item() ** .5
                    print('Epoch: {:d}, Batch: {:d}/{:d}, |Param|: {:.6f}, |GParam|: {:.2f}, lr: {:.2E}, loss: {:.4f}, recon: {:.4f}, kl: {:.4f}, log_det: {:.4f}'.format(
                        epoch, batch_idx, len(dataloader), param_norm, gparam_norm, optimizer.param_groups[0]['lr'], (train_nll + train_kl + train_log_det) / n,
                        train_nll / n_frames, train_kl / n_frames, train_log_det / n, (train_nll + train_kl) / n_frames
                    ))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                model.zero_grad()
        train_loss = np.mean(losses)
        if scheduler is not None:
            scheduler.step(train_loss)

if __name__ == '__main__':
    C = 3
    train_dset = ToyDataset(*synthetic_data(C=C, num_points=1500), max_k=20)
    test_dset = ToyDataset(*synthetic_data(C=C, num_points=50), max_k=20)

    train_unsupervised(train_dset)

    '''model = train_supervised(train_dset)
    train_loader = DataLoader(train_dset, batch_size=10)
    test_loader = DataLoader(test_dset, batch_size=10)
    train_acc, train_pred = predict(model, train_loader)
    test_acc, test_pred = predict(model, test_loader)
    print('Train acc: {:.2f}'.format(train_acc))
    print('Test acc: {:.2f}'.format(test_acc))'''
