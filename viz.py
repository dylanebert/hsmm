import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from hsmm import optimal_map
import argparse
import data
import pickle


def viz_parameters(model):
    means = model.gaussian_means.detach().cpu().numpy()
    cov = model.gaussian_cov.detach().cpu().numpy()
    init = model.initial_log_probs(None).detach().cpu().numpy()[:, np.newaxis]
    trans = model.transition_log_probs(None).detach().cpu().numpy()
    lengths = model.length_log_probs(None).detach().cpu().numpy()

    init = np.exp(init)
    trans = np.exp(trans)
    lengths = np.exp(lengths)

    fig = plt.figure()

    ax1 = fig.add_subplot(321)
    ax1.imshow(means, interpolation='nearest', cmap=cm.Blues, vmin=0, vmax=1)
    ax1.set_xlabel('Feature mean')
    ax1.set_ylabel('State')
    ax1.set_xticks(np.arange(means.shape[1]))
    ax1.set_yticks(np.arange(means.shape[1]))
    ax1.set_title('Gaussian means')

    ax2 = fig.add_subplot(322)
    ax2.imshow(cov, interpolation='nearest', cmap=cm.Blues, vmin=0, vmax=1)
    ax2.set_xticks(np.arange(cov.shape[0]))
    ax2.set_yticks(np.arange(cov.shape[0]))
    ax2.set_title('Gaussian covariances')

    ax3 = fig.add_subplot(323)
    ax3.imshow(lengths.transpose(), interpolation='nearest', cmap=cm.Blues, vmin=0)
    ax3.set_xlabel('Length')
    ax3.set_ylabel('State')
    ax3.set_xticks(np.arange(lengths.shape[0]))
    ax3.set_yticks(np.arange(lengths.shape[1]))
    ax3.set_title('Poisson rates')

    ax4 = fig.add_subplot(324)
    ax4.imshow(trans, interpolation='nearest', cmap=cm.Blues, vmin=0, vmax=1)
    ax4.set_xlabel('From state')
    ax4.set_ylabel('To state')
    ax4.set_xticks(np.arange(trans.shape[1]))
    ax4.set_yticks(np.arange(trans.shape[0]))
    ax4.set_title('Transition rates')

    ax5 = fig.add_subplot(325)
    ax5.imshow(init, interpolation='nearest', cmap=cm.Blues, vmin=0, vmax=1)
    ax5.set_ylabel('State')
    ax5.set_xticks([])
    ax5.set_yticks(np.arange(init.shape[0]))
    ax5.set_title('Initial rates')

    plt.tight_layout()
    plt.show()


def viz_state_seq(model, dataset, remap=True):
    features = dataset[0]['features'].unsqueeze(0)
    lengths = dataset[0]['lengths'].unsqueeze(0)
    if 'valid_classes' in dataset[0]:
        valid_classes = dataset[0]['valid_classes'].unsqueeze(0)
    else:
        valid_classes = [torch.LongTensor(list(range(dataset.n_classes))).to(features.device)]

    N_ = lengths.max().item()
    features = features[:, :N_, :]

    pred_spans = model.viterbi(features, lengths, valid_classes_per_instance=valid_classes, add_eos=True)
    pred_labels = data.spans_to_labels(pred_spans)
    pred_labels_trim = model.trim(pred_labels, lengths)

    if 'spans' in dataset[0]:
        gold_spans = dataset[0]['spans'].unsqueeze(0)
        gold_spans = gold_spans[:, :N_]
        gold_labels = data.spans_to_labels(gold_spans)
        gold_labels_trim = model.trim(gold_labels, lengths)

        if remap:
            pred_remapped, mapping = optimal_map(pred_labels_trim[0], gold_labels_trim[0], valid_classes[0])
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
    else:
        pred = pred_labels_trim[0].detach().cpu().numpy()[np.newaxis, :]
        pred = np.tile(pred, (N_ // 10, 1))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.imshow(pred)
        ax1.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='model meta file name', type=str, required=True)
    args = parser.parse_args()

    with open('models/{}.p'.format(args.name), 'rb') as f:
        meta = pickle.load(f)
    args = meta['args']
    dset = data.dataset_from_args(args, 'test')
    model = torch.load(meta['model'])

    viz_state_seq(model, dset, remap=(args.model == 'unsupervised'))
    viz_parameters(model)
