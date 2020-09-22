import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from hsmm import SemiMarkovModule, optimal_map
import argparse
from tqdm import tqdm
import os
from datetime import datetime
import glob
import data
from data import labels_to_spans, spans_to_labels
import pickle

random.seed(a=0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def untrained_model(args, n_classes):
    n_dim = train_dset[0]['features'].shape[-1]
    assert n_dim == test_dset[0]['features'].shape[-1]

    model = SemiMarkovModule(args, n_classes, n_dim, args.max_k)
    if device == torch.device('cuda'):
        model.cuda()

    return model

def train_supervised(args, train_dset, n_classes):
    n_dim = train_dset[0]['features'].shape[-1]
    assert n_dim == test_dset[0]['features'].shape[-1]

    model = SemiMarkovModule(args, n_classes, n_dim, args.max_k)
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

def train_unsupervised(args, train_dset, test_dset, n_classes):
    train_loader = DataLoader(train_dset, batch_size=10)
    test_loader = DataLoader(test_dset, batch_size=10)

    n_dim = train_dset[0]['features'].shape[-1]
    assert n_dim == test_dset[0]['features'].shape[-1]

    model = SemiMarkovModule(args, n_classes, n_dim, args.max_k)
    if device == torch.device('cuda'):
        model.cuda()
    model.initialize_gaussian(train_dset.features, train_dset.lengths)
    optimizer = torch.optim.Adam(model.parameters(), model.learning_rate)

    if args.overrides is not None:
        for override in args.overrides:
            assert override in ['mean', 'cov', 'init', 'trans', 'lengths'], override
        train_features = []
        train_labels = []
        for i in range(len(train_dset)):
            sample = train_dset[i]
            train_features.append(sample['features'])
            train_labels.append(sample['labels'])
        model.initialize_supervised(train_features, train_labels, overrides=args.overrides)

    model.train()
    best_loss = 1e9
    k = 0
    patience = 5
    best_model = type(model)(args, n_classes, n_dim, args.max_k)
    if device == torch.device('cuda'):
        best_model.cuda()
    if args.debug:
        deep_debug(model, test_dset, 0)
    for epoch in range(args.epochs):
        losses = []
        for batch in train_loader:
            features = batch['features']
            lengths = batch['lengths']
            if 'valid_classes' in batch:
                valid_classes = batch['valid_classes']
            else:
                valid_classes = None
            N_ = lengths.max().item()
            features = features[:, :N_, :]
            loss, _ = model.log_likelihood(features, lengths, valid_classes)
            loss = -loss
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            model.zero_grad()
        if 'labels' in train_dset[0]:
            train_remap_acc, train_action_remap_acc, train_pred = predict(model, train_loader, remap=True)
            test_remap_acc, test_action_remap_acc, test_pred = predict(model, test_loader, remap=True)
            print('epoch: {}, avg loss: {:.4f}, train acc: {:.2f}, test acc: {:.2f}, train step acc: {:.2f}, test step acc: {:.2f}'.format(
                epoch, np.mean(losses), train_remap_acc, test_remap_acc, train_action_remap_acc, test_action_remap_acc))

            if args.debug:
                deep_debug(model, test_dset, epoch+1)
        else:
            print('epoch: {}, avg_loss: {:.4f}'.format(epoch, np.mean(losses)))
        if np.mean(losses) < best_loss:
            best_loss = np.mean(losses) - 1e-4
            best_model.load_state_dict(model.state_dict())
            k = 0
        else:
            k += 1
            print('Loss didn\'t improve for {} epoch(s)'.format(k))
            if k == patience:
                print('Stopping')
                break

    return best_model

def predict(model, dataloader, remap=True):
    items = []
    token_match, remap_match, token_total = 0, 0, 0
    action_match, action_remap_match, action_total = 0, 0, 0
    for batch in dataloader:
        features = batch['features']
        lengths = batch['lengths']
        gold_spans = batch['spans']
        if 'valid_classes' in batch:
            valid_classes = batch['valid_classes']
        else:
            valid_classes = None

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
            pred_remapped = pred_remapped.to(device)
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
            action_remap_match += (gold_labels_trim[i] == pred_remapped)[gold_labels_trim[i] != 0].sum().item()
            token_total += pred_labels_trim[i].size(0)
            action_total += (gold_labels_trim[i] != 0).sum()
    accuracy = 100. * token_match / token_total
    remapped_accuracy = 100. * remap_match / token_total
    action_accuracy = 100. * action_match / action_total
    action_remapped_accuracy = 100. * action_remap_match / action_total
    if remap:
        return remapped_accuracy, action_remapped_accuracy, items
    else:
        return accuracy, action_accuracy, items

def deep_debug(model, test_dset, epoch):
    features = test_dset[0]['features'].unsqueeze(0)
    lengths = test_dset[0]['lengths'].unsqueeze(0)
    valid_classes = None

    transition_log_probs = model.transition_log_probs(valid_classes)
    emission_log_probs = model.emission_log_probs(features, valid_classes)
    initial_log_probs = model.initial_log_probs(valid_classes)
    length_log_probs = model.length_log_probs(valid_classes)

    print('Epoch {0}\n---------------------'.format(epoch))
    print('Transition\n', transition_log_probs, '\n')
    print('Emission\n', emission_log_probs, '\n')
    print('Initial\n', initial_log_probs, '\n')
    print('Lengths\n', length_log_probs, '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    data.add_args(parser)
    SemiMarkovModule.add_args(parser)
    parser.add_argument('--model', choices=['untrained', 'supervised', 'unsupervised'], default='unsupervised')
    parser.add_argument('--overrides', nargs='+')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=999)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device)

    train_dset, test_dset = data.dataset_from_args(args)
    assert train_dset.n_classes == test_dset.n_classes
    n_classes = train_dset.n_classes

    if args.model == 'untrained':
        model = untrained_model(args, n_classes)
    elif args.model == 'supervised':
        model = train_supervised(args, train_dset, n_classes)
    elif args.model == 'unsupervised':
        model = train_unsupervised(args, train_dset, test_dset, n_classes)

    if args.debug:
        print('Debug mode - not saving')
    else:
        modelpath = 'models/{}.pt'.format(datetime.now().strftime('%Y%m%d-%H%M%S'))
        torch.save(model, modelpath)

        metapath = 'models/{}_{}_{}.p'.format(args.dataset, args.model, args.suffix)
        meta = {
            'model': modelpath,
            'args': args
        }
        with open(metapath, 'wb+') as f:
            pickle.dump(meta, f)

    if 'labels' in train_dset[0]:
        train_loader = DataLoader(train_dset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dset, batch_size=args.batch_size)
        train_acc, train_action_acc, train_pred = predict(model, train_loader, remap=(args.model == 'unsupervised'))
        test_acc, test_action_acc, test_pred = predict(model, test_loader, remap=(args.model == 'unsupervised'))
        print('Train acc: {:.2f}; step: {:.2f}'.format(train_acc, train_action_acc))
        print('Test acc: {:.2f}; step: {:.2f}'.format(test_acc, test_action_acc))
