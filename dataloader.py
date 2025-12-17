import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from pathlib import Path
import sys
import os

EXP_NAME = sys.argv[1]
DATA_DIR = os.path.join('data', EXP_NAME)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
ROSBAG_PATH = Path(os.path.join('data', sys.argv[2]))
OUTPUT_FILE = os.path.join(DATA_DIR, 'data')
SCAN_TOPIC = '/autodrive/roboracer_1/lidar'
ODOM_TOPIC = '/tf'

typestore = get_typestore(Stores.ROS2_HUMBLE)

def process_bag():
    scans = []
    poses = []
    
    print(f"Reading {ROSBAG_PATH}...")
    last_pose = None
    
    with AnyReader([ROSBAG_PATH], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic in [SCAN_TOPIC, ODOM_TOPIC]]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            if connection.topic == ODOM_TOPIC:
                px = msg.transforms[0].transform.translation.x
                py = msg.transforms[0].transform.translation.y
                
                qx = msg.transforms[0].transform.rotation.x
                qy = msg.transforms[0].transform.rotation.y
                qz = msg.transforms[0].transform.rotation.z
                qw = msg.transforms[0].transform.rotation.w
                
                siny_cosp = 2 * (qw * qz + qx * qy)
                cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                theta = np.arctan2(siny_cosp, cosy_cosp)
                
                last_pose = np.array([px, py, theta])
                
            elif connection.topic == SCAN_TOPIC:
                if last_pose is None:
                    continue
                
                ranges = np.array(msg.ranges[1::4])
                
                ranges[np.isinf(ranges)] = 30.0
                ranges[np.isnan(ranges)] = 30.0
                
                scans.append(ranges)
                poses.append(last_pose)

    poses = np.array(poses)
    scans = np.array(scans)
    
    print(f"Poses: {poses.shape} | Scans: {scans.shape}")

    data_record = np.concatenate([poses, scans], axis=1)
    
    print(f"Saving to {OUTPUT_FILE}...")
    np.savez(OUTPUT_FILE, data_record=data_record)
    print("Done!")

if __name__ == "__main__":
    process_bag()