import pandas as pd
import re

participants = {
    'train': ['1_1a', '2_2a', '5_1c', '6_2c',
              '7_1a', '8_2a', '9_1b', '10_2b', '11_1c', '12_2c',
              '13_1a', '14_2a', '15_1b', '16_2b'],
    'dev': ['17_1c', '18_2c'],
    'test': ['3_1b', '4_2b']
}

participant_lookup = {}
for k, v in participants.items():
    for elem in v:
        participant_lookup[elem] = k

obj_names = ['Knife', 'Banana', 'Apple', 'Fork', 'Plant', 'Book', 'Spoon', 'Bowl', 'Cup',
             'Lamp', 'Ball', 'Bear', 'Toy', 'Doll', 'RightHand', 'LeftHand', 'Head', 'Dinosaur']


def load_nbc_data():
    import glob
    fnames = glob.glob('data/nbc_slim/*.p')
    data = []
    for fname in fnames:
        rows = pd.read_pickle(fname)
        session = re.findall(r'\d+_\d\w_task\d', fname)[0]
        participant = '_'.join(session.split('_')[:2])
        rows['session'] = session
        rows['type'] = participant_lookup[participant]
        data.append(rows)
    data = pd.concat(data).fillna(0)
    return data


if __name__ == '__main__':
    data = load_nbc_data()
    print(data['posX'].index)
