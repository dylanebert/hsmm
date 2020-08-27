import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from hsmm import SemiMarkovModule, labels_to_spans, spans_to_labels
import argparse
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import nbc_data

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
        if self.valid_classes is None:
            return {
                'labels': self.labels[index],
                'features': self.features[index],
                'lengths': self.lengths[index],
                'spans': spans
            }
        else:
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

def optimal_map(pred, true, possible):
    assert all(l in possible for l in pred) and all(l in possible for l in true)
    table = np.zeros((len(possible), len(possible)))
    labels = possible.detach().cpu().numpy()
    for i, label in enumerate(labels):
        mask = true == label
        for j, l in enumerate(labels):
            table[i, j] = (pred[mask] == l).sum()
    best_true, best_pred = linear_sum_assignment(-table)
    mapping = {labels[p]: labels[g] for p, g in zip(best_pred, best_true)}
    remapped = pred.clone()
    remapped.apply_(lambda label: mapping[label])
    return remapped, mapping

def untrained_model(n_classes):
    parser = argparse.ArgumentParser()
    SemiMarkovModule.add_args(parser)
    args = parser.parse_args()
    model = SemiMarkovModule(args, n_classes, n_classes, allow_self_transitions=True)

    return model

def train_supervised(train_dset, n_classes):
    parser = argparse.ArgumentParser()
    SemiMarkovModule.add_args(parser)
    args = parser.parse_args()
    model = SemiMarkovModule(args, n_classes, n_classes, allow_self_transitions=True)

    train_features = []
    train_labels = []
    for i in range(len(train_dset)):
        sample = train_dset[i]
        train_features.append(sample['features'])
        train_labels.append(sample['labels'])
    model.fit_supervised(train_features, train_labels)

    return model

def train_unsupervised(train_loader, test_loader, n_classes, epochs=25):
    parser = argparse.ArgumentParser()
    SemiMarkovModule.add_args(parser)
    args = parser.parse_args()
    args.sm_max_span_length = 20

    model = SemiMarkovModule(args, n_classes, n_classes, allow_self_transitions=True)

    s = next(iter(train_loader))
    features = s['features']
    lengths = s['lengths']
    model.initialize_gaussian(features, lengths)

    optimizer = torch.optim.Adam(model.parameters(), model.learning_rate)

    model.train()
    for epoch in range(epochs):
        losses = []
        for batch in train_loader:
            features = batch['features']
            lengths = batch['lengths']
            valid_classes = None#batch['valid_classes']
            N_ = lengths.max().item()
            features = features[:, :N_, :]
            loss, _ = model.log_likelihood(features, lengths, valid_classes)
            loss = -loss
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            model.zero_grad()
        train_acc, train_remap_acc, train_action_acc, train_action_remap_acc, train_pred = predict(model, train_loader)
        test_acc, test_remap_acc, test_action_acc, test_action_remap_acc, test_pred = predict(model, test_loader)
        print('epoch: {}, avg loss: {:.4f}, train acc: {:.2f}, test acc: {:.2f}, train action acc: {:.2f}, test action acc: {:.2f}'.format(
            epoch, np.mean(losses), train_remap_acc, test_remap_acc, train_action_remap_acc, test_action_remap_acc))

    return model

def predict(model, dataloader):
    items = []
    token_match, remap_match, token_total = 0, 0, 0
    action_match, action_remap_match = 0, 0
    for batch in dataloader:
        features = batch['features']
        lengths = batch['lengths']
        gold_spans = batch['spans']
        valid_classes = None#batch['valid_classes']

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
            if valid_classes is None:
                valid_classes_ = torch.LongTensor(np.array(list(range(model.n_classes)), dtype=int))
            else:
                valid_classes_ = valid_classes[i]
            pred_remapped, mapping = optimal_map(pred_labels_trim[i], gold_labels_trim[i], valid_classes_)
            item = {
                'length': lengths[i].item(),
                'gold_spans': gold_spans[i],
                'pred_spans': pred_spans[i],
                'gold_labels': gold_labels[i],
                'pred_labels': pred_labels[i],
                'gold_labels_trim': gold_labels_trim[i],
                'pred_labels_trim': pred_labels_trim[i],
                'pred_remap_trim': pred_remapped,
                'mapping': mapping
            }
            items.append(item)
            token_match += (gold_labels_trim[i] == pred_labels_trim[i]).sum().item()
            remap_match += (gold_labels_trim[i] == pred_remapped).sum().item()
            action_match += (gold_labels_trim[i] == pred_labels_trim[i])[gold_labels_trim[i] != 0].sum().item()
            action_remap_match += (gold_labels_trim[i] == pred_labels_trim[i])[gold_labels_trim[i] != 0].sum().item()
            token_total += pred_labels_trim[i].size(0)
    accuracy = 100. * token_match / token_total
    remapped_accuracy = 100. * remap_match / token_total
    action_accuracy = 100. * action_match / token_total
    action_remapped_accuracy = 100. * action_remap_match / token_total
    return accuracy, remapped_accuracy, action_accuracy, action_remapped_accuracy, items

if __name__ == '__main__':
    #train_dset = ToyDataset(*synthetic_data(C=3, num_points=150), max_k=20)
    #test_dset = ToyDataset(*synthetic_data(C=3, num_points=50), max_k=20)

    train_dset = ToyDataset(*nbc_data.get_onehot_dataset('train'), max_k=180)
    test_dset = ToyDataset(*nbc_data.get_onehot_dataset('test'), max_k=180)

    n_classes = train_dset[0]['features'].size(1)

    train_loader = DataLoader(train_dset, batch_size=10)
    test_loader = DataLoader(test_dset, batch_size=10)

    #model = untrained_model(n_classes)
    #model = train_supervised(train_dset, n_classes)
    model = train_unsupervised(train_loader, test_loader, n_classes)

    train_acc, train_remap_acc, train_action_acc, train_action_remap_acc, train_pred = predict(model, train_loader)
    test_acc, test_remap_acc, test_action_acc, test_action_remap_acc, test_pred = predict(model, test_loader)
    print('Train acc: {:.2f}'.format(train_acc))
    print('Train remap acc: {:.2f}'.format(train_remap_acc))
    print('Train action acc: {:.2f}'.format(train_action_acc))
    print('Train action remap acc: {:.2f}'.format(train_action_remap_acc))
    print('Test acc: {:.2f}'.format(test_acc))
    print('Test remap acc: {:.2f}'.format(test_remap_acc))
    print('Test action acc: {:.2f}'.format(test_action_acc))
    print('Test action remap acc: {:.2f}'.format(test_action_remap_acc))
