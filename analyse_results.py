import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

EXP_NAME = sys.argv[1]
MODEL_NAME = "best"
DATA_DIR = os.path.join('data', EXP_NAME)
DATA_FILE = os.path.join(DATA_DIR, 'train_data.npz')
CONF_FILE = os.path.join(DATA_DIR, 'train_data.json')
OUT_DIR = os.path.join('results', EXP_NAME)
TEST_SIZE = 4000
DEVICE = "cpu"

from model import Local_INN, PositionalEncoding
from utils.utils import ConfigJSON, DataProcessor

def main():
    device = torch.device(DEVICE)
    print(f"--- Analyzing {EXP_NAME} ---")
    c = ConfigJSON()
    if not os.path.exists(CONF_FILE):
        print(f"Error: {CONF_FILE} not found. Run preprocess_data.py first.")
        return
    c.load_file(CONF_FILE)
    
    norm_x = c.d['normalization_x']
    norm_y = c.d['normalization_y']
    norm_lidar = c.d['normalization_laser']
    
    raw_data = np.load(DATA_FILE)
    if 'data_record' in raw_data:
        total_data = raw_data['data_record']
    else:
        total_data = raw_data['arr_0']
        
    test_data = total_data[-TEST_SIZE:]
    print(f"Loaded {len(test_data)} test samples.")
    
    model = Local_INN(device=device)
    weights_path = f'results/{EXP_NAME}/{EXP_NAME}_model_{MODEL_NAME}.pt'
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    dp = DataProcessor()
    pe_cond = PositionalEncoding(L=1).to(device)
    pe_pose = PositionalEncoding(L=10).to(device)
    
    errors = []
    points = []
    
    print("Running Inference...")
    with torch.no_grad():
        for i in range(len(test_data)):
            row = test_data[i]
            
            gt_scan = row[60:330]
            gt_pose_raw = row[330:333]

            y_scan = torch.from_numpy(gt_scan).float().unsqueeze(0).to(device)
            
            cond_raw = torch.from_numpy(gt_pose_raw).float().unsqueeze(0).to(device)
            c_encoded = pe_cond(cond_raw)
            
            z_vae = model.vae.encoder(y_scan)
            
            z_input = torch.zeros((1, 60), device=device)
            z_input[:, :54] = z_vae
            
            x_pred, _ = model.reverse(z_input, c_encoded)

            pred_x_norm = pe_pose.batch_decode(x_pred[:, 0], x_pred[:, 1]).item()
            pred_y_norm = pe_pose.batch_decode(x_pred[:, 2], x_pred[:, 3]).item()
            
            real_x_pred = dp.de_normalize(pred_x_norm, norm_x)
            real_y_pred = dp.de_normalize(pred_y_norm, norm_y)
            
            real_x_gt = dp.de_normalize(gt_pose_raw[0], norm_x)
            real_y_gt = dp.de_normalize(gt_pose_raw[1], norm_y)
            
            err = np.sqrt((real_x_pred - real_x_gt)**2 + (real_y_pred - real_y_gt)**2)
            
            errors.append(err)
            points.append({
                'idx': i,
                'err': err,
                'scan': gt_scan,
                'pose': [real_x_gt, real_y_gt]
            })

    errors = np.array(errors)
    sorted_indices = np.argsort(errors)
    
    best_indices = sorted_indices[:3]
    worst_indices = sorted_indices[-3:]
    
    print(f"\nAvg Error: {np.mean(errors):.4f} m")
    print(f"Max Error: {np.max(errors):.4f} m")
    
    plot_lidar_comparison(best_indices, worst_indices, points, dp, norm_lidar)
    plot_error_map(points)
    
def plot_lidar_comparison(best_idx, worst_idx, points, dp, norm_lidar):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), subplot_kw={'projection': 'polar'})
    
    theta = np.linspace(-135 * np.pi/180, 135 * np.pi/180, 270)
    
    def plot_row(row_axs, indices, title_prefix):
        for k, idx in enumerate(indices):
            ax = row_axs[k]
            p = points[idx]

            scan_m = dp.de_normalize(p['scan'], norm_lidar)
            
            ax.plot(theta, scan_m, label='Lidar')
            ax.set_title(f"{title_prefix} #{k+1}\nError: {p['err']:.2f}m")
            ax.set_ylim(0, 10)
            ax.grid(True)

    plot_row(axs[0], best_idx, "BEST")
    plot_row(axs[1], worst_idx, "WORST")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "lidar_analysis.png"))
    print("Saved lidar_analysis.png")
    plt.show()

def plot_error_map(points):
    x = [p['pose'][0] for p in points]
    y = [p['pose'][1] for p in points]
    err = [p['err'] for p in points]
    
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(x, y, c=err, cmap='jet', s=20)
    plt.colorbar(sc, label='Localization Error (m)')
    plt.title("Spatial Error Map (Red = High Error)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(OUT_DIR, "error_map.png"))
    print("Saved error_map.png")
    plt.show()

if __name__ == "__main__":
    main()