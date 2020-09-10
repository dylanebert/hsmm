import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from hsmm import SemiMarkovModule
import argparse
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import os
from datetime import datetime
import glob
import data
from data import labels_to_spans, spans_to_labels

random.seed(a=0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda')

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
    remapped = pred.cpu().clone()
    remapped.apply_(lambda label: mapping[label])
    return remapped.to(device), mapping

def untrained_model(n_classes, max_k):
    parser = argparse.ArgumentParser()
    SemiMarkovModule.add_args(parser)
    args = parser.parse_args(['--sm_max_span_length', str(max_k)])

    model = SemiMarkovModule(args, n_classes, n_classes, allow_self_transitions=True)
    if device == torch.device('cuda'):
        model.cuda()

    return model

def train_supervised(train_dset, n_classes, max_k):
    parser = argparse.ArgumentParser()
    SemiMarkovModule.add_args(parser)
    args = parser.parse_args(['--sm_max_span_length', str(max_k)])

    model = SemiMarkovModule(args, n_classes, n_classes, allow_self_transitions=True)
    if device == torch.device('cuda'):
        model.cuda()

    train_features = []
    train_labels = []
    for i in range(len(train_dset)):
        sample = train_dset[i]
        train_features.append(sample['features'])
        train_labels.append(sample['labels'])
    model.fit_supervised(train_features, train_labels)

    return model

def train_unsupervised(train_dset, test_dset, n_classes, max_k, epochs=999):
    parser = argparse.ArgumentParser()
    SemiMarkovModule.add_args(parser)
    args = parser.parse_args(['--sm_max_span_length', str(max_k)])

    train_loader = DataLoader(train_dset, batch_size=10)
    test_loader = DataLoader(test_dset, batch_size=10)

    model = SemiMarkovModule(args, n_classes, n_classes, allow_self_transitions=False)
    if device == torch.device('cuda'):
        model.cuda()
    model.initialize_gaussian(train_dset.features, train_dset.lengths)
    optimizer = torch.optim.Adam(model.parameters(), model.learning_rate)

    #supervised overrides for debugging
    '''train_features = []
    train_labels = []
    for i in range(len(train_dset)):
        sample = train_dset[i]
        train_features.append(sample['features'])
        train_labels.append(sample['labels'])
    model.initialize_supervised(train_features, train_labels, overrides=['mean', 'cov'])'''

    model.train()
    best_loss = 1e9
    for epoch in range(epochs):
        losses = []
        for batch in train_loader:
            features = batch['features']
            lengths = batch['lengths']
            valid_classes = batch['valid_classes']
            N_ = lengths.max().item()
            features = features[:, :N_, :]
            loss, _ = model.log_likelihood(features, lengths, valid_classes)
            loss = -loss
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            model.zero_grad()
        train_remap_acc, train_action_remap_acc, train_pred = predict(model, train_loader, remap=True)
        test_remap_acc, test_action_remap_acc, test_pred = predict(model, test_loader, remap=True)
        print('epoch: {}, avg loss: {:.4f}, train acc: {:.2f}, test acc: {:.2f}, train step acc: {:.2f}, test step acc: {:.2f}'.format(
            epoch, np.mean(losses), train_remap_acc, test_remap_acc, train_action_remap_acc, test_action_remap_acc))
        if np.mean(losses) < best_loss:
            best_loss = np.mean(losses) - 1e-7
        else:
            print('Loss didn\'t improve, stopping')
            break

    return model

def predict(model, dataloader, remap=True):
    items = []
    token_match, remap_match, token_total = 0, 0, 0
    action_match, action_remap_match = 0, 0
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
            if valid_classes is None:
                valid_classes_ = torch.LongTensor(np.array(list(range(model.n_classes)), dtype=int)).to(device)
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
    if remap:
        return remapped_accuracy, action_remapped_accuracy, items
    else:
        return accuracy, action_accuracy, items

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    data.add_args(parser)
    parser.add_argument('--model', choices=['untrained', 'supervised', 'unsupervised'], default='unsupervised')
    parser.add_argument('--epochs', type=int, default=999)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    train_dset, test_dset = data.dataset_from_args(args)
    n_classes = train_dset[0]['features'].size(1)

    if args.model == 'untrained':
        model = untrained_model(n_classes, args.max_k)
    elif args.model == 'supervised':
        model = train_supervised(train_dset, n_classes, args.max_k)
    elif args.model == 'unsupervised':
        model = train_unsupervised(train_dset, test_dset, n_classes, args.max_k, epochs=args.epochs)

    if args.debug:
        print('Debug mode - not saving')
    else:
        modelpath = 'models/{}_{}'.format(args.dataset, args.model)
        version = glob.glob(modelpath + '*')
        if version == []:
            savepath = modelpath + '0.pt'
        else:
            version_num = int(version[-1][len(modelpath):].replace('.pt', ''))
            version_num += 1
            savepath = '{}{}.pt'.format(modelpath, version_num)
        print('Saving model to {}'.format(savepath))
        torch.save(model, savepath)

    train_loader = DataLoader(train_dset, batch_size=10)
    test_loader = DataLoader(test_dset, batch_size=10)
    train_acc, train_action_acc, train_pred = predict(model, train_loader, remap=(args.model == 'unsupervised'))
    test_acc, test_action_acc, test_pred = predict(model, test_loader, remap=(args.model == 'unsupervised'))
    print('Train acc: {:.2f}; step: {:.2f}'.format(train_acc, train_action_acc))
    print('Test acc: {:.2f}; step: {:.2f}'.format(test_acc, test_action_acc))
