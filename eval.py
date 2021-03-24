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
    def __init__(self, hsmm_wrapper, type='dev'):
        self.hsmm_wrapper = hsmm_wrapper
        self.input_module = hsmm_wrapper.input_module
        self.type = type
        self.get_eval()

    def get_eval(self):
        if self.load():
            return
        self.predictions = self.hsmm_wrapper.predictions[self.type]
        self.rle()
        self.get_questions()
        self.save()

    def load(self):
        return False
        fpath = NBC_ROOT + 'cache/eval/{}.json'.format(self.hsmm_wrapper.fname)
        if not os.path.exists(fpath):
            return False
        with open(fpath) as f:
            serialized = json.load(f)
        self.questions = serialized['questions']
        self.answers = serialized['answers']
        print('loaded eval from {}'.format(fpath))
        return True

    def save(self):
        fpath = NBC_ROOT + 'cache/eval/{}.json'.format(self.hsmm_wrapper.fname)
        serialized = {'questions': self.questions, 'answers': self.answers}
        with open(fpath, 'w+') as f:
            json.dump(serialized, f)
        print('saved eval to {}'.format(fpath))

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

    def get_questions(self):
        questions = []
        for label in np.unique(np.array(self.predictions[0])):
            for i in range(10):
                sample = self.sample(label)
                question = []
                for j in range(4):
                    entry = sample[j]
                    start_step = int(entry['start_step'])
                    end_step = int(entry['end_step'])
                    start_timestamp = float(entry['start_timestamp'])
                    end_timestamp = float(entry['end_timestamp'])
                    qlabel = entry['label']
                    session = entry['session']
                    oddmanout = j == 3
                    question.append((oddmanout, session, start_timestamp, end_timestamp, start_step, end_step, qlabel))
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
        for i, (key, steps) in enumerate(self.input_module.steps[self.type].items()):
            session_start_step = steps[0][0]
            predictions = self.predictions[i]
            prev_idx = 0
            k = 1
            for i in range(1, len(predictions)):
                prev_label = predictions[prev_idx]
                label = predictions[i]
                if label != prev_label:
                    start_step = steps[prev_idx][0]
                    end_step = steps[i-1][-1]
                    start_timestamp = (start_step - session_start_step) / 90.
                    end_timestamp = (end_step - session_start_step) / 90.
                    if prev_label not in rle_dict:
                        rle_dict[prev_label] = []
                    item = {
                        'start_step': start_step,
                        'end_step': end_step,
                        'start_timestamp': start_timestamp,
                        'end_timestamp': end_timestamp,
                        'session': key[0],
                        'label': prev_label
                    }
                    if i - prev_idx > 10:
                        rle_dict[prev_label].append(item)
                    prev_idx = i
        self.rle_dict = rle_dict

    def report(self):
        correct, sum = 0, 0
        for i in range(len(self.answers)):
            answer = self.answers[i]
            question = self.questions[i]
            gold = (np.array([q[0] for q in question[1]]) == True).argmax()
            if answer == gold:
                correct += 1
            sum += 1
        print(correct / sum)

if __name__ == '__main__':
    from hsmm_wrapper import HSMMWrapper
    hsmm_wrapper = HSMMWrapper('hsmm_max_objs_nclasses=2')
    eval = OddManOut(hsmm_wrapper)
    eval.report()
