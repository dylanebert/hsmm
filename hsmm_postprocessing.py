import numpy as np
import sys
import os
from hsmm_wrapper import HSMMWrapper

assert 'NBC_ROOT' in os.environ, 'set NBC_ROOT'
NBC_ROOT = os.environ['NBC_ROOT']
sys.path.append(NBC_ROOT)


def merge_similar_states(hsmm_wrapper):
    predictions = hsmm_wrapper.predictions
    model = hsmm_wrapper.model
    means = model.gaussian_means.detach().cpu().numpy()
    merge_map = {}
    for i in range(means.shape[0] - 1):
        for j in range(i + 1, means.shape[0]):
            dist = np.linalg.norm(means[i] - means[j])
            if dist < 5e-1:
                if i in merge_map:
                    merge_map[i].append(j)
                else:
                    assert j not in merge_map
                    added = False
                    for k, v in merge_map.items():
                        if i in v:
                            if j not in merge_map[k]:
                                merge_map[k].append(j)
                            added = True
                            break
                    if not added:
                        merge_map[i] = [j]

    merge_pointers = {}
    for k, v in merge_map.items():
        for val in v:
            merge_pointers[val] = k
    for i in range(means.shape[0]):
        if i not in merge_pointers:
            merge_pointers[i] = i
    merge_pointers = dict(sorted(merge_pointers.items(), key=lambda item: item[0]))

    values = np.unique(list(merge_pointers.values()))
    label_mapping = {}
    for v in values:
        label_mapping[v] = len(label_mapping)

    labels = {}
    for k, v in merge_pointers.items():
        labels[k] = label_mapping[v]

    for type in ['train', 'dev', 'test']:
        for i in range(len(predictions[type])):
            predictions[type][i] = [labels[x] for x in predictions[type][i]]
    hsmm_wrapper.predictions = predictions


if __name__ == '__main__':
    hsmm_wrapper = HSMMWrapper('hsmm_engineered')
    merge_similar_states(hsmm_wrapper)
