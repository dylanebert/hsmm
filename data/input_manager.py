from scipy.spatial.transform import Rotation as R
import pandas as pd


def subsample(data):
    indices = list(range(0, len(data), 9))
    data = data.iloc[indices]
    return data


def compute_depth(data):
    data = data.sort_index()
    idx = pd.IndexSlice
    head_pos = data.loc[:, idx[['posX', 'posY', 'posZ'], 'Head']].to_numpy()
    head_rot = data.loc[:, idx[['rotX', 'rotY', 'rotZ', 'rotW'], 'Head']].to_numpy()
    head_rot = R.from_quat(head_rot)
    head_yaw = head_rot.as_euler('xyz', degrees=True)
    head_yaw[:, [0, 2]] = 0
    head_yaw = R.from_euler('xyz', head_yaw, degrees=True)

    for hand in ['LeftHand', 'RightHand']:
        hand_pos = data.loc[:, idx[['posX', 'posY', 'posZ'], hand]].to_numpy()
        hand_rel = head_yaw.apply(hand_pos - head_pos, inverse=True)
        data.loc[:, idx['horizon', hand]] = hand_rel[:, 2]
        data.loc[:, idx['height', hand]] = hand_rel[:, 1]
        data.loc[:, idx['depth', hand]] = hand_rel[:, 0]

    return data


if __name__ == '__main__':
    import nbc_bridge
    data = nbc_bridge.load_nbc_data()
    data = data[data['session'] == '1_1a_task1']
    data = compute_depth(data)
    print(data)
