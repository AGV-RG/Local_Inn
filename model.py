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
DATAFILE = os.path.join(DATA_DIR, 'train_data.npz')
NORMFILE = os.path.join(DATA_DIR, 'train_data.json')
COLS = 333
TEST_DATA = 3000

CONTINUE_TRAINING = 0
TRANSFER_TRAINING = 0
TRANSFER_EXP_NAME = ''

BATCHSIZE = 500
LR = 5e-4
COND_NOISE = [0.2, 0.2, 15 / 180 * np.pi] 
SCAN_NOISE = 0.005
COND_DIM = 6

def main():
    device = torch.device('cpu')
    print(f"Running on device: {device}")
    writer = SummaryWriter('results/tensorboard/' + EXP_NAME)
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = torch.from_numpy(data).type('torch.FloatTensor').to(device)
        def __len__(self):
            return len(self.data)
        def __getitem__(self, index):
            return self.data[index]
    print("EXP_NAME", EXP_NAME)
    if not os.path.exists('results/' + EXP_NAME + '/'):
            os.makedirs('results/' + EXP_NAME + '/')

    if os.path.exists(DATAFILE):
        print("Loading pre-processed data...")
        total_data = np.load(DATAFILE)['data_record']
        c = ConfigJSON()
        c.load_file(NORMFILE)
        norm_x_range = c.d['normalization_x'][0] 
        norm_y_range = c.d['normalization_y'][0]
        dp = DataProcessor()
    else:
        print("Preprocess Data First")
        return
        
    assert total_data.shape[1] == COLS, f"Data Shape Mismatch! Expected {COLS} columns, got {total_data.shape[1]}"
    print(f"Total Data Shape Verified: {total_data.shape}")

    train_data = total_data[0:total_data.shape[0] - TEST_DATA]
    test_data = total_data[total_data.shape[0] - TEST_DATA:]
    print(f"Train Data Shape: {train_data.shape}")
    print(f"Test Data Shape: {test_data.shape}")
    
    p_encoding_t = PositionalEncoding(L=1).to(device)

    cond_noise = np.array(COND_NOISE)
    cond_noise[0] /= norm_x_range
    cond_noise[1] /= norm_y_range
    cond_noise[2] /= np.pi * 2
    cond_noise = torch.from_numpy(cond_noise).type('torch.FloatTensor').to(device)
    
    train_set = Dataset(train_data)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCHSIZE, shuffle=True)
    test_set = Dataset(test_data)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCHSIZE, shuffle=False)
    l1_loss = torch.nn.L1Loss()
    
    trainer = Trainer(EXP_NAME, 500, 0.0001, device,
                      LR, [300], 0.05, 'exponential',
                      False, 3, 0.99, 0)
    model = Local_INN(device=device)
    model.to(device)
    
    if CONTINUE_TRAINING:
        model = trainer.continue_train_load(model, path='results/' + EXP_NAME + '/')

    current_lr = LR
    optimizer = torch.optim.Adam(model.trainable_parameters, lr=current_lr)
    optimizer.add_param_group({"params": model.cond_net.parameters(), "lr": current_lr})
    optimizer.add_param_group({"params": model.vae.encoder.parameters(), "lr": current_lr})
    optimizer.add_param_group({"params": model.vae.decoder.parameters(), "lr": current_lr})

    n_hypo = 20
    epoch_time = 0
    
    while(not trainer.is_done()):
        epoch = trainer.epoch
        epoch_info = np.zeros(7)
        epoch_info[3] = epoch
        epoch_time_start = time.time()
        
        trainer_lr = trainer.get_lr()
        if trainer_lr != current_lr:
            current_lr = trainer_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
                
        model.train()
        model.vae.encoder.train()
        model.vae.decoder.train()
        
        for data in train_loader:
            optimizer.zero_grad()
            
            x_hat_gt = data[:, :60]
            y_gt = data[:, 60:330]
            cond = data[:, 330:333]
            
            cond += torch.zeros_like(cond, device=device).normal_(0., 1.) * cond_noise
            cond = p_encoding_t.forward(cond.round(decimals=1))
            
            y_hat_vae = torch.zeros_like(x_hat_gt, device=device)
            y_hat_vae[:, :-6] = model.vae.encoder.forward(y_gt)
            y_hat_inn, _ = model(x_hat_gt, cond)
            y_inn = model.vae.decoder.forward(y_hat_inn[:, :-6])
            
            vae_kl_loss = model.vae.encoder.kl * 0.0001
            inn_recon_loss = l1_loss(y_inn, y_gt)
            y_hat_inn_loss = l1_loss(y_hat_inn[:, :-6], y_hat_vae[:, :-6])
            loss_forward = vae_kl_loss + inn_recon_loss + y_hat_inn_loss
            epoch_info[0] += loss_forward.item()
            epoch_info[1] += inn_recon_loss.item()
            
            loss_forward.backward(retain_graph=True)
            
            y_hat_vae[:, -6:] = 0
            x_hat_0, _ = model.reverse(y_hat_vae, cond)
            loss_reverse = l1_loss(x_hat_0[:, :12], x_hat_gt[:, :12])
            
            batch_size = y_gt.shape[0]
            
            z_samples = torch.randn(n_hypo, batch_size, 6, device=device)
            
            y_hat = y_hat_vae[None, :, :54].repeat(n_hypo, 1, 1)
            y_hat_z_samples = torch.cat((y_hat, z_samples), dim=2).view(-1, 60)
            cond_rep = cond[None].repeat(n_hypo, 1, 1).view(-1, COND_DIM) # Uses corrected COND_DIM
            x_hat_i = model.reverse(y_hat_z_samples, cond_rep)[0].view(n_hypo, batch_size, 60)
            x_hat_i_loss = torch.mean(torch.min(torch.mean(torch.abs(x_hat_i[:, :, :12] - x_hat_gt[:, :12]), dim=2), dim=0)[0])
            loss_reverse += x_hat_i_loss
            epoch_info[2] += loss_reverse.item()
            
            loss_reverse.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 8)
            optimizer.step()
            
        epoch_info[:3] /= len(train_loader)

        model.eval()
        model.vae.encoder.eval()
        model.vae.decoder.eval()
        epoch_posit_err = []
        epoch_orient_err = []
        with torch.no_grad():
            for data in test_loader:
                x_hat_gt = data[:, :60]
                x_gt = data[:, 330:333]
                cond = data[:, 330:333]
                y_gt = data[:, 60:330]
                
                cond += torch.zeros_like(cond, device=device).normal_(0., 1.) * cond_noise
                cond = p_encoding_t.forward(cond.round(decimals=1))
                
                y_hat_vae = torch.zeros_like(x_hat_gt, device=device)
                y_hat_vae[:, :-6] = model.vae.encoder.forward(y_gt)
                x_hat_0, _ = model.reverse(y_hat_vae, cond)
                
                pred_x = p_encoding_t.batch_decode(x_hat_0[:, 0], x_hat_0[:, 1])
                pred_y = p_encoding_t.batch_decode(x_hat_0[:, 2], x_hat_0[:, 3])

                real_x = dp.de_normalize(pred_x, c.d['normalization_x'])
                real_y = dp.de_normalize(pred_y, c.d['normalization_y'])
                
                gt_x = dp.de_normalize(x_gt[:, 0], c.d['normalization_x'])
                gt_y = dp.de_normalize(x_gt[:, 1], c.d['normalization_y'])

                pos_err = torch.sqrt((real_x - gt_x)**2 + (real_y - gt_y)**2)
                epoch_posit_err.append(pos_err)

                pred_theta = p_encoding_t.batch_decode_even(x_hat_0[:, 4], x_hat_0[:, 5])
                pred_theta_rad = pred_theta * 2 * np.pi
                gt_theta_rad   = x_gt[:, 2] * 2 * np.pi

                ang_err = torch.abs(gt_theta_rad - pred_theta_rad)
                ang_err = torch.min(ang_err, 2*np.pi - ang_err)
                epoch_orient_err.append(ang_err)
            
            if len(epoch_posit_err) > 0:
                avg_pos_err = torch.cat(epoch_posit_err).median().item()
                avg_ang_err = torch.cat(epoch_orient_err).median().item()
            else:
                avg_pos_err = 0.0
                avg_ang_err = 0.0

            epoch_info[4] = avg_pos_err
            epoch_info[5] = avg_ang_err

        epoch_time = (time.time() - epoch_time_start)
        remaining_time = (trainer.max_epoch - epoch) * epoch_time / 3600

        model, return_text, _ = trainer.step(model, epoch_info, 0)
        if return_text == 'instable':
            optimizer = torch.optim.Adam(model.trainable_parameters, lr=current_lr)
            optimizer.add_param_group({"params": model.cond_net.parameters(), "lr": current_lr})
            optimizer.add_param_group({"params": model.vae.encoder.parameters(), "lr": current_lr})
            optimizer.add_param_group({"params": model.vae.decoder.parameters(), "lr": current_lr})
        
        writer.add_scalar("INN/0_forward", epoch_info[0], epoch)
        writer.add_scalar("INN/1_recon", epoch_info[1], epoch)
        writer.add_scalar("INN/2_reverse", epoch_info[2], epoch)
        writer.add_scalar("INN/Err_Pos_m", epoch_info[4], epoch)
        writer.add_scalar("INN/Err_Ang_rad", epoch_info[5], epoch)
        writer.add_scalar("INN/5_LR", current_lr, epoch)
        
        text_print = "Ep {} | Fwd {:.4f} | Rev {:.4f} | PosErr {:.2f}m | AngErr {:.2f}r | {:.1f}h left".format(
            epoch, epoch_info[0], epoch_info[2], epoch_info[4], epoch_info[5], remaining_time)
        print(text_print)
        with open('results/' + EXP_NAME + '/' + EXP_NAME + '.txt', "a") as tgt:
            tgt.writelines(text_print + '\n')
    writer.flush()
        
if __name__ == '__main__':
    main()