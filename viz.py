import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from hsmm import SemiMarkovModule
import glob
import argparse
from main import optimal_map, labels_to_spans, spans_to_labels, ToyDataset, synthetic_data
import nbc_data

def viz_parameters(model):
    means = model.gaussian_means.detach().cpu().numpy()
    cov = model.gaussian_cov.detach().cpu().numpy()
    init = model.init_logits.detach().cpu().numpy()[:, np.newaxis]
    trans = model.transition_logits.detach().cpu().numpy()

    init = np.exp(init)
    trans = np.exp(trans)

    fig = plt.figure()

    ax1 = fig.add_subplot(221)
    ax1.imshow(means, interpolation='nearest', cmap=cm.Blues, vmin=0, vmax=1)
    ax1.set_xlabel('Feature mean')
    ax1.set_ylabel('State')
    ax1.set_xticks(np.arange(means.shape[0]))
    ax1.set_yticks(np.arange(means.shape[0]))
    ax1.set_title('Gaussian means')

    ax2 = fig.add_subplot(222)
    ax2.imshow(cov, interpolation='nearest', cmap=cm.Blues, vmin=0, vmax=1)
    ax2.set_xticks(np.arange(cov.shape[0]))
    ax2.set_yticks(np.arange(cov.shape[0]))
    ax2.set_title('Gaussian covariances')

    ax3 = fig.add_subplot(223)
    ax3.imshow(trans, interpolation='nearest', cmap=cm.Blues, vmin=0, vmax=1)
    ax3.set_xlabel('From state')
    ax3.set_ylabel('To state')
    ax3.set_xticks(np.arange(trans.shape[0]))
    ax3.set_yticks(np.arange(trans.shape[1]))
    ax3.set_title('Transition rates')

    ax4 = fig.add_subplot(224)
    ax4.imshow(init, interpolation='nearest', cmap=cm.Blues, vmin=0, vmax=1)
    ax4.set_ylabel('State')
    ax4.set_xticks([])
    ax4.set_yticks(np.arange(init.shape[0]))
    ax4.set_title('Initial rates')

    plt.tight_layout()
    plt.show()

def viz_state_seq(model, dataset, remap=True):
    features = dataset[0]['features'].unsqueeze(0)
    lengths = dataset[0]['lengths'].unsqueeze(0)
    gold_spans = dataset[0]['spans'].unsqueeze(0)
    valid_classes = dataset[0]['valid_classes'].unsqueeze(0)

    batch_size = features.size(0)
    N_ = lengths.max().item()
    features = features[:, :N_, :]
    gold_spans = gold_spans[:, :N_]

    pred_spans = model.viterbi(features, lengths, valid_classes_per_instance=valid_classes, add_eos=True)
    gold_labels = spans_to_labels(gold_spans)
    pred_labels = spans_to_labels(pred_spans)

    gold_labels_trim = model.trim(gold_labels, lengths)
    pred_labels_trim = model.trim(pred_labels, lengths)

    if remap:
        pred_remapped, mapping = optimal_map(pred_labels_trim[0], gold_labels_trim[0], valid_classes[0])
        print(mapping)
        gold = gold_labels_trim[0].detach().cpu().numpy()[np.newaxis, :]
        pred = pred_remapped.detach().cpu().numpy()[np.newaxis, :]
    else:
        gold = gold_labels_trim[0].detach().cpu().numpy()[np.newaxis, :]
        pred = pred_labels_trim[0].detach().cpu().numpy()[np.newaxis, :]

    gold = np.tile(gold, (N_ // 10, 1))
    pred = np.tile(pred, (N_ // 10, 1))

    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax1.imshow(gold)
    ax1.axis('off')

    ax2 = fig.add_subplot(212)
    ax2.imshow(pred)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['toy', 'nbc-like'], default='nbc-like')
    parser.add_argument('--model', choices=['background', 'supervised', 'unsupervised'], default='supervised')
    args = parser.parse_args()

    model_fpaths = sorted(list(glob.glob('models/{}_{}*'.format(args.dataset, args.model))))
    print('Loading model from path {}'.format(model_fpaths[-1]))
    model = torch.load(model_fpaths[-1])

    if args.dataset == 'toy':
        dataset = ToyDataset(*synthetic_data(C=3, num_points=150), max_k=20)
    elif args.dataset == 'nbc-like':
        dataset = ToyDataset(*nbc_data.annotations_dataset('test'), max_k=20)

    viz_state_seq(model, dataset, remap=(args.model == 'unsupervised'))
    viz_parameters(model)
