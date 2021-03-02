import numpy as np
import pandas as pd
import random
import os
import sys
import json

assert 'NBC_ROOT' in os.environ, 'set NBC_ROOT'
NBC_ROOT = os.environ['NBC_ROOT']
sys.path.append(NBC_ROOT)
import config

class OddManOut:
    def __init__(self, nbc_wrapper, hsmm_wrapper, type='dev'):
        self.args = hsmm_wrapper.args
        self.nbc_wrapper = nbc_wrapper
        self.hsmm_wrapper = hsmm_wrapper
        self.type = type
        self.get_eval()

    def get_eval(self):
        if self.try_load_cached():
            return
        self.keys = list(self.nbc_wrapper.nbc.steps[self.type].keys())
        self.steps = list(self.nbc_wrapper.nbc.steps[self.type].values())
        self.sessions = np.array([key[0] for key in self.keys])
        self.indices = self.hsmm_wrapper.sequences[self.type][2]
        self.predictions = self.hsmm_wrapper.predictions[self.type]
        self.rle()
        self.get_questions()
        self.cache()

    def evaluate(self, fill_unknown=True):
        correct = 0
        total = 0
        for question, answer in zip(self.questions, self.answers):
            gold = -1
            for i in range(4):
                if question[1][i][0]:
                    gold = i
                    break
            if fill_unknown and answer == -1:
                answer = 0
            if gold == answer:
                correct += 1
            total += 1
        print(correct / total)

    def try_load_cached(self):
        savefile = config.find_savefile(self.args, 'eval')
        if savefile is None:
            return False
        question_path = NBC_ROOT + 'cache/eval/{}_questions.json'.format(savefile)
        answer_path = NBC_ROOT + 'cache/eval/{}_answers.json'.format(savefile)
        with open(question_path) as f:
            self.questions = json.load(f)
        with open(answer_path) as f:
            self.answers = json.load(f)
        print('loaded cached eval')
        return True

    def cache(self):
        savefile = config.generate_savefile(self.args, 'eval')
        question_path = NBC_ROOT + 'cache/eval/{}_questions.json'.format(savefile)
        answer_path = NBC_ROOT + 'cache/eval/{}_answers.json'.format(savefile)
        with open(question_path, 'w+') as f:
            json.dump(self.questions, f)
        with open(answer_path, 'w+') as f:
            json.dump(self.answers, f)
        print('cached eval')

    def update_answers(self):
        savefile = config.find_savefile(self.args, 'eval')
        assert savefile is not None
        answer_path = NBC_ROOT + 'cache/eval/{}_answers.json'.format(savefile)
        with open(answer_path, 'w+') as f:
            json.dump(self.answers, f)
        print('updated answers')

    def get_questions(self):
        questions = []
        for label in np.unique(np.array(self.predictions[0])):
            for i in range(10):
                sample = self.sample()
                question = []
                for j in range(4):
                    idx, k = sample[j]
                    start_step, end_step = int(self.steps[idx][0]), int(self.steps[idx+k][-1])
                    session = self.sessions[idx]
                    session_start_step = self.keys[(self.sessions == self.keys[idx][0]).argmax()][1]
                    start_timestamp = (start_step - session_start_step) / 90.
                    end_timestamp = (end_step - session_start_step) / 90.
                    oddmanout = j == 3
                    question.append((oddmanout, session, start_timestamp, end_timestamp, start_step, end_step))
                random.shuffle(question)
                questions.append((int(label), question))
        random.shuffle(questions)
        self.questions = questions
        self.answers = [None] * len(questions)

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

    def rle(self):
        rle_dict = {}
        for predictions, indices in zip(self.predictions, self.indices):
            prev_label = predictions[0]
            k = 1
            for i in range(1, len(predictions)):
                label = predictions[i]
                idx = indices[i]
                if label != prev_label:
                    if label not in rle_dict:
                        rle_dict[label] = []
                    rle_dict[label].append((idx, k))
                    prev_label = label
                    k = 1
                else:
                    k += 1
        self.rle_dict = rle_dict
