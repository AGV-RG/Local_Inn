import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from trainer import Trainer
from LocalINN import Local_INN
from PositionalEncoding import PositionalEncoding
from VariationalAutoEncoder import VAE
from utils.utils import ConfigJSON, DataProcessor

EXP_NAME = sys.argv[1]
DATA_DIR = os.path.join('data', EXP_NAME)
DATAFILE = os.path.join(DATA_DIR, 'data.npz')
OUTPUT_FILE = os.path.join(DATA_DIR, 'train_data.npz')

device = torch.device('cpu')

if not os.path.exists(DATAFILE):
    raise FileNotFoundError(f"Could not find data file at: {DATAFILE}")
    
total_data = np.load(DATAFILE)['data_record']

dp = DataProcessor()
c = ConfigJSON()

total_data[:, 0], c.d['normalization_x'] = dp.data_normalize(total_data[:, 0])
total_data[:, 1], c.d['normalization_y'] = dp.data_normalize(total_data[:, 1])

norm_x_range = c.d['normalization_x'][0]
norm_y_range = c.d['normalization_y'][0]

total_data[:, 2] = dp.two_pi_warp(total_data[:, 2])

theta_max, theta_min = 2 * np.pi, 0.0
c.d['normalization_theta'] = [theta_max, theta_min]

lidar_max, lidar_min = 30.0, 0.0
c.d['normalization_laser'] = [lidar_max, lidar_min]
total_data[:, 3:] = dp.runtime_normalize(total_data[:, 3:], [lidar_max, lidar_min])

c.save_file(DATA_DIR + '/' + 'train_data.json')

print("Encoding data...")
pe = PositionalEncoding(L=10).to(device)

temp_tensor = torch.from_numpy(total_data[:, :3]).float().to(device)
encoded_pose = pe(temp_tensor).cpu().numpy()

total_data = np.concatenate([encoded_pose, total_data[:, 3:], total_data[:, :3]], axis=1)

np.savez(OUTPUT_FILE, data_record=total_data)