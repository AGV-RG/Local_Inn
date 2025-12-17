import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, L=10):
        super(PositionalEncoding, self).__init__()
        self.L = L
        val_list = []
        for l in range(L):
            val_list.append(2.0 ** l)
        self.register_buffer('val_list', torch.tensor(val_list).unsqueeze(0))
        self.register_buffer('pi', torch.tensor(3.14159265358979323846))

    def encode(self, x):
        return torch.sin(x * self.val_list * self.pi), torch.cos(x * self.val_list * self.pi)
    
    def encode_even(self, x):
        return torch.sin(x * self.val_list * self.pi * 2), torch.cos(x * self.val_list * self.pi * 2)
    
    def forward(self, batch):
        batch_encoded_list = []
        for ind in range(3):
            if ind == 2:
                encoded_ = self.encode_even(batch[:, ind, None])
            else:
                encoded_ = self.encode(batch[:, ind, None])
            batch_encoded_list.append(encoded_[0])
            batch_encoded_list.append(encoded_[1])
        batch_encoded = torch.stack(batch_encoded_list)
        return batch_encoded.transpose(0, 1).transpose(1, 2).reshape((batch_encoded.shape[1], self.L * batch_encoded.shape[0]))

    def batch_decode(self, sin_value, cos_value):
        atan2_value = torch.atan2(sin_value, cos_value) / (self.pi)
        atan2_value = torch.abs(atan2_value)
        return atan2_value

    def batch_decode_even(self, sin_value, cos_value):
        atan2_value = torch.atan2(sin_value, cos_value) / self.pi / 2
        sub_zero_inds = torch.where(atan2_value < 0)
        atan2_value[sub_zero_inds] = atan2_value[sub_zero_inds] + 1
        atan2_value[torch.where(torch.abs(atan2_value - 1) < 0.001)] = 0
        return atan2_value