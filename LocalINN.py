import torch
import torch.nn as nn
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, ConditionNode
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom
from VariationalAutoEncoder import VAE
from PositionalEncoding import PositionalEncoding

class Local_INN(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.inn_input_dim = int(60)
        self.cond_input_dim = 6
        self.cond_out_dim = 12 
        self.model = self.build_inn()

        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        for param in self.trainable_parameters:
            param.data = 0.05 * torch.randn_like(param)
        
        self.cond_net = self.subnet_cond(self.cond_out_dim)
        self.pose_encoding = PositionalEncoding(L=10)
        self.cond_encoding = PositionalEncoding(L=1)
        self.vae = VAE()

    def subnet_cond(self, c_out):
        return nn.Sequential(nn.Linear(self.cond_input_dim, 256), nn.ReLU(),
                             nn.Linear(256, self.cond_out_dim))
    
    def build_inn(self):

        def subnet_fc(c_in, c_out):
            return nn.Sequential(
                nn.Linear(c_in, 1024), 
                nn.ReLU(),
                nn.Linear(1024, c_out)
            )

        nodes = [InputNode(self.inn_input_dim, name='input')]
        cond = ConditionNode(self.cond_out_dim, name='condition')
        for k in range(6):
            nodes.append(Node(nodes[-1],
                              GLOWCouplingBlock,
                              {'subnet_constructor': subnet_fc, 'clamp': 2.0},
                              conditions=cond,
                              name=f'coupling_{k}'))
            
            nodes.append(Node(nodes[-1],
                              PermuteRandom,
                              {'seed': k},
                              name=f'permute_{k}'))

        nodes.append(OutputNode(nodes[-1], name='output'))
        
        return ReversibleGraphNet(nodes + [cond], verbose=False).to(self.device)

    def forward(self, x, cond):
        return self.model(x, self.cond_net(cond))

    def reverse(self, y_rev, cond):
        return self.model(y_rev, self.cond_net(cond), rev=True)