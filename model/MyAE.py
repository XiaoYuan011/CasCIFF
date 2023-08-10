import torch.nn as nn
import torch.nn.functional as F
class AE(nn.Module):
    def __init__(self,input_dim,h_dim,z_dim):
        super(AE, self).__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.set_layers()
    def set_layers(self):
        self.encoder=nn.Sequential(
            nn.Linear(self.input_dim, self.h_dim) ,
            nn.ReLU(True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(True),
            nn.Linear(self.h_dim, self.z_dim),
            nn.Tanh()
        )
        self.decoder=nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(self.h_dim, self.input_dim),
            nn.ReLU(True)
        )
    def forward(self,x):
        z=self.encoder(x)
        z_recon=self.decoder(z)
        loss=F.mse_loss(x,z_recon)
        return loss,z