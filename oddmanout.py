from data import input_manager
import numpy as np
import os
import time


def prepare_question_commands(actions, group, questions_per_action=10):
    labels = np.array(actions['label'].unique())
    for label, rows in actions.groupby('label'):
        for i in range(questions_per_action):
            same = rows.sample(n=3, replace=True)
            odd_label = np.random.choice(labels[~(labels == label)])
            odd = actions[actions['label'] == odd_label].sample(n=1)
            odd_idx = np.random.choice(range(4))
            same_idx = np.arange(4)
            same_idx = same_idx[~(same_idx == odd_idx)]
            qs = {}
            for i in range(3):
                qs['abcd'[same_idx[i]]] = 'https://plunarlabcit.services.brown.edu/eval/hands_height_depth/{}.mp4'.format(same.index.values[i])
            qs['abcd'[odd_idx]] = 'https://plunarlabcit.services.brown.edu/eval/hands_height_depth/{}.mp4'.format(odd.index.values[0])
            cmd = "insert into oddmanout (a, b, c, d, correct, response, 'group') values('{0}', '{1}', '{2}', '{3}', {4}, -1, {5});".format(qs['a'], qs['b'], qs['c'], qs['d'], odd_idx, group)
            print(cmd)


def prepare_video_commands(data, actions, dname):
    NBC_ROOT = os.environ['NBC_ROOT']
    cmds = []
    for i, action in actions.iterrows():
        dst_dir = f'{NBC_ROOT}/eval/{dname}'
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        session = action['session']
        start_step = action['start_step']
        end_step = action['end_step']
        start_timestamp = data.loc[(session, start_step)]['timestamp'].values[0]
        end_timestamp = data.loc[(session, end_step)]['timestamp'].values[0]
        src = f'${{NBC_ROOT}}/videos/{session}.mp4'
        dst = f'${{NBC_ROOT}}/eval/{dname}/{i}.mp4'
        start_string = time.strftime('%H:%M:%S.{}'.format('{:.3f}'.format(start_timestamp % 1000 / 1000.).replace('0.', '').replace('1.', '')), time.gmtime(start_timestamp / 1000.))
        dur = end_timestamp - start_timestamp
        duration_string = time.strftime('%H:%M:%S.{}'.format('{:.3f}'.format(dur % 1000 / 1000.).replace('0.', '').replace('1.', '')), time.gmtime(dur / 1000.))
        cmd = f'ffmpeg -i {src} -ss {start_string} -t {duration_string} -async 1 -strict -2 -f mp4 {dst}'
        cmds.append(cmd)
    with open(NBC_ROOT + '/vidcmds.sh', 'w+') as f:
        f.write('#!/bin/bash\n')
        f.write('\n'.join(cmds))


if __name__ == '__main__':
    data = input_manager.load_cached('energy')
    actions = input_manager.load_cached('actions_with_cluster_labels_16')
    prepare_video_commands(data, actions, 'hands_height_depth')
    actions = actions[actions['type'] == 'dev']
    prepare_question_commands(actions, 1, questions_per_action=2)
