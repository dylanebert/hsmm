import numpy as np
import pandas as pd
import random
import os
import sys

assert 'NBC_ROOT' in os.environ, 'set NBC_ROOT'
NBC_ROOT = os.environ['NBC_ROOT']
sys.path.append(NBC_ROOT)
import config

class OddManOut:
    def __init__(self, nbc_wrapper, hsmm_wrapper, type='dev'):
        keys = list(nbc_wrapper.nbc.steps[type].keys())
        self.steps = list(nbc_wrapper.nbc.steps[type].values())
        self.sessions = np.array([key[0] for key in keys])
        self.indices = hsmm_wrapper.sequences[type][2]
        self.predictions = hsmm_wrapper.predictions[type]
        self.rle()

    def get_eval(self):
        questions = []
        for label in np.unique(np.array(self.predictions[0])):
            for i in range(10):
                sample = self.sample()
                question = []
                for j in range(4):
                    idx, k = sample[j]
                    session = self.sessions[idx]
                    start_step, end_step = int(self.steps[idx][0]), int(self.steps[idx+k][-1])
                    question.append((session, start_step, end_step))
                questions.append((int(label), question))
        return questions

    def rle(self):
        rle_dict = {}
        for label in np.unique(np.array(self.predictions[0])):
            rle_dict[label] = []
        for predictions, indices in zip(self.predictions, self.indices):
            prev_label = predictions[0]
            k = 1
            for i in range(1, len(predictions)):
                label = predictions[i]
                idx = indices[i]
                if label != prev_label:
                    rle_dict[label].append((idx, k))
                    prev_label = label
                    k = 1
                else:
                    k += 1
        self.rle_dict = rle_dict

    def sample(self, primary_label=None):
        labels = np.array(list(self.rle_dict.keys()))
        assert len(labels >= 2)
        if primary_label == None:
            primary_label = random.choice(labels.tolist())
        secondary_label = random.choice(labels[labels != primary_label].tolist())
        choices = []
        for i in range(3):
            choices.append(random.choice(self.rle_dict[primary_label]))
        choices.append(random.choice(self.rle_dict[secondary_label]))
        return choices
