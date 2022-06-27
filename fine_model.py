import torch
import torch.nn as nn
import numpy as np
import pdb
from imresize import isotropic_gaussian_kernel

                
class HSIchannel(nn.Module):
    def __init__(self, opt):
        super(HSIchannel, self).__init__()        
                     
        factor = opt.upscale_factor
        kernel_size = 21
        kernel = isotropic_gaussian_kernel(np.array([kernel_size, kernel_size]), np.array([factor, factor]), factor/2.) 
        kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).type(torch.cuda.FloatTensor)
        if kernel_size % 2 == 1:
            pad = int((kernel_size - 1) / 2.)
        else:
            pad = int((kernel_size - factor) / 2.)            

        self.padding = nn.ReplicationPad2d(pad)               

        self.Conv = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=factor)        
        self.Conv.weight = nn.Parameter(kernel)
              
               
    def forward(self, x):

        x = x.permute(1,0,2,3)
        x = self.padding(x)
        x = self.Conv(x)        
        x = x.permute(1,0,2,3)
        
        return x
                 
                    
class FineNet(nn.Module):

    def __init__(self, opt):
        super(FineNet, self).__init__()
        self.ReLU = nn.ReLU(inplace=True) 
        wn = lambda x: torch.nn.utils.weight_norm(x)         
        self.Conv1 = wn(nn.Conv2d(34, 192, 3, 1, 1))      
        self.Conv2 = wn(nn.Conv2d(192, 192, 3, 1, 1))        
        self.Conv3 = wn(nn.Conv2d(192, 31, 3, 1, 1))   
#        self.Conv4 = nn.Conv2d(64, 31, 3, 1, 1)
        
        self.hsi = HSIchannel(opt)

    def forward(self, x, y, z):

        out = torch.cat([x,y], 1)
        out = self.Conv1(out) 
        out = self.Conv2(self.ReLU(out))           
        out = self.Conv3(self.ReLU(out))  
#        out = self.Conv4(self.ReLU(out))                 
        out = out + x 
                  
        return  out, self.hsi(out) 