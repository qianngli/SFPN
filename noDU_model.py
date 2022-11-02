import torch
import torch.nn as nn
import pdb
from patch import calc_padding
import torch.nn.functional as F                                        
        
class Unit(nn.Module):
    def __init__(self, wn, n_feats, kernel_size = 3, padding = 1, bias = True, act=nn.ReLU(inplace=True)):
        super(Unit, self).__init__()        
        
        m = []
        m.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size, padding, bias)))
        m.append(act)
        m.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size, padding, bias)))
   
        self.m = nn.Sequential(*m)
    
    def forward(self, x):
        
        x = self.m(x) + x    
        return x 

class Aggregate(nn.Module):
    def __init__(self, wn, n_feats, stride = 5, patchsize = 5):
        super(Aggregate, self).__init__()
        
        self.stride = stride
        self.patchsize = patchsize
        
        self.speFusion = wn(nn.Conv2d(n_feats*4, n_feats, kernel_size=3, padding=1, bias=True)) 
        self.gamma = nn.Parameter(torch.ones(3))   
                
    def forward(self, img, corr):

        index = corr[0]
        values = corr[1]
                
        padtop, padbottom, padleft, padright = calc_padding(img.shape[2:], self.patchsize, self.stride, padding=None)    
        xpad = F.pad(img, pad=(padleft, padright, padtop, padbottom))     
        
        B, C, W, H = xpad.shape

        Z = F.unfold(xpad, kernel_size=self.patchsize, stride=self.stride)
        B, C_kh_kw, L = Z.size()
        Z = Z.permute(0, 2, 1)
        Z = Z.view(B, L, -1, self.patchsize, self.patchsize)   
        
        fusionZ = torch.zeros(B, L, C*4, self.patchsize, self.patchsize).cuda()

        row = int(W / self.stride) 
        col = int(H / self.stride) 
              
        for i in range(row):
            for j in range(col):
                k = col*i + j
                patch = Z[:,k,:,:,:] 
                
                patch_index = []
                
                for n in range(B):	
                    selected_patch = torch.index_select(Z, 1, index[k][n])
                    
                    channel = selected_patch[n]
               	                                
                    patch_index.append(torch.cat([patch[n], channel[0]* values[k][n][0], channel[1]* values[k][n][1], 
                	            channel[2]* values[k][n][2]], 0).unsqueeze(0))

                fusionZ[:,k,:,:,:] = torch.cat(patch_index, 0) 
                	             
        fusionZ = fusionZ.view(B, L, -1)
        fusionZ = fusionZ.permute(0, 2, 1)
        fusionZ = F.fold(fusionZ, (W, H), kernel_size=self.patchsize, stride=self.stride)
        fusionZ = fusionZ[:,:,padtop:W-padbottom, padleft:H-padright]
        
        local_F = self.speFusion(fusionZ)       
                                                                      
        return local_F            

class FASplit(nn.Module):
    def __init__(self, wn, n_feats, scale, stride, patchsize, kernel_size=3, padding=1, bias=True, act=nn.ReLU(inplace=True)): 
        super(FASplit, self).__init__()
               
        self.unit_hsi = wn(nn.Conv2d(n_feats, n_feats, kernel_size, padding, bias))      
        self.unit_rgb = wn(nn.Conv2d(n_feats, n_feats, kernel_size, padding, bias))  
        self.aggregate = Aggregate(wn, n_feats, stride, patchsize)

        self.reduce  = wn(nn.Conv2d(n_feats*2, n_feats, kernel_size=1, bias=True))                              
        self.n_feats = n_feats
                
    def forward(self, hsi, rgb, corr):

        n_feats = int(self.n_feats//2)
        hsi_out = torch.cat([hsi[:,0:n_feats,:,:], rgb[:,0:n_feats,:,:]], 1)
        rgb_out = torch.cat([hsi[:,n_feats:self.n_feats,:,:], rgb[:,n_feats:self.n_feats,:,:]], 1)
	
        hsi_out = self.unit_hsi(hsi_out)
        rgb_out = self.unit_rgb(rgb_out)
            	
        hsi_rgb_out = self.reduce(torch.cat([hsi_out, rgb_out], 1)) 
        hsi_out = self.aggregate(hsi_rgb_out, corr)
    	    	
        return hsi_out + hsi, hsi_out + rgb
    	
	
class bottleneck(nn.Module):
    def __init__(self,  wn, n_feats, scale, n_module, kernel_size=1, padding=1 ,bias=True):
        super(bottleneck, self).__init__()    

        self.hsi_fusion = wn(nn.Conv2d(n_feats*n_module, n_feats, kernel_size, bias=bias))                 
        self.rgb_fusion = wn(nn.Conv2d(n_feats*n_module, n_feats, kernel_size, bias=bias))
        
        self.gamma = nn.Parameter(torch.ones(2))         
        self.fusion = wn(nn.Conv2d(n_feats*2, n_feats, kernel_size=3, padding=padding, bias=bias)) 
 
                                  
    def forward(self, hsi, skip_hsi, rgb, skip_rgb):
        hsi = self.hsi_fusion(hsi) + skip_hsi
        
        rgb = self.rgb_fusion(rgb) +  skip_rgb     
        hsi = torch.cat([self.gamma[0]*hsi, self.gamma[1]*rgb], 1)
        hsi = self.fusion(hsi) 
    
        return hsi

class Head(nn.Sequential):
    def __init__(self, wn, input_feats, output_feats, kernel_size, padding=1, bias=True):

        m = []
        m.append(wn(nn.Conv2d(input_feats, output_feats, kernel_size, padding=1, bias=True)))
                    
        super(Head, self).__init__(*m)  

class inter_module(nn.Module):
    def __init__(self, type, wn, n_feats, scale,  stride, patchsize, kernel_size = 3, padding = 1, bias = True, act=nn.ReLU(inplace=True)):
        super(inter_module, self).__init__()
        self.act = act

        self.unit_hsi_1 = Unit(wn, n_feats)
        self.unit_rgb_1 = Unit(wn, n_feats)
        
        self.unit_hsi_2 = Unit(wn, n_feats)        
        self.unit_rgb_2 = Unit(wn, n_feats)
                
        
        self.fusion = FASplit(wn, n_feats, scale, stride, patchsize)
    
    def forward(self, data):

        hsi = data[0]
        rgb = data[1]
        corr = data[2]

        out_hsi = self.unit_hsi_1(hsi)                       
        out_rgb = self.unit_rgb_1(rgb)
   	
        out_hsi, out_rgb = self.fusion(out_hsi, out_rgb, corr)
    	
        out_hsi = self.unit_hsi_2(out_hsi) 
        out_rgb = self.unit_rgb_2(out_rgb)
    	    	
        return  out_hsi + hsi, out_rgb + rgb

class CoarseNet(nn.Module):
    def __init__(self, args):
        super(CoarseNet, self).__init__()
        
        scale = args.upscale_factor
        n_feats = args.n_feats          
        kernel_size = 3
        self.n_module = args.n_module
        stride = args.stride
        patchsize = args.patchsize
        
        wn = lambda x: torch.nn.utils.weight_norm(x) 
        	                                    
        self.hsi_head = nn.Conv2d(3, n_feats, kernel_size, padding=1, bias=True) 
        
        self.speFusion = wn(nn.Conv2d(n_feats*2, n_feats, kernel_size=1, bias=True))  
        self.gamma_speContext = nn.Parameter(torch.ones(2)) 
        self.rgb_head = nn.Conv2d(3, n_feats, kernel_size, padding=1, bias=True) 
 
        inter_body = [
                      inter_module(args.type, wn, n_feats, scale, scale*stride, scale*patchsize
                  ) for _ in range(self.n_module)
        ]
        self.inter_body =  nn.Sequential(*inter_body) 
        
        self.gamma_hsi = nn.Parameter(torch.ones(self.n_module))
        self.gamma_rgb = nn.Parameter(torch.ones(self.n_module))                      
        
        self.bottleneck = bottleneck(wn, n_feats, scale, self.n_module)
        self.feaFusion = wn(nn.Conv2d(n_feats*2, n_feats, kernel_size=1, bias=True))       
        self.gamma_feaContext = nn.Parameter(torch.ones(2))              

        self.hsi_rgb_end = Head(wn, n_feats, 1, kernel_size)        

        self.spe_head = nn.Conv2d(1, n_feats, kernel_size, padding=1, bias=True)                		            

        self.nearest = nn.Upsample(scale_factor=scale, mode='nearest')
                                            
    def forward(self, index, hsi, neigbor, rgb, corr, spe_context=None, feats_context=None):

        hsi = torch.cat([neigbor[:,0,:,:].unsqueeze(1), hsi.unsqueeze(1), neigbor[:,1,:,:].unsqueeze(1)], 1)
        hsi = self.nearest(hsi)
        
        hsi =  self.hsi_head(hsi) 
        rgb = self.rgb_head(rgb) 
        
        skipRGB = rgb
        skipHSI = hsi 
                     
        if index > 0:
            spe_context = self.spe_head(spe_context)
            hsi = torch.cat([self.gamma_speContext[0]*hsi, self.gamma_speContext[1]*spe_context], 1)
            hsi = self.speFusion(hsi)
                                                           
        out_hsi = []
        out_rgb = []
                
        for i in range(self.n_module):        
            hsi, rgb = self.inter_body[i]([hsi, rgb, corr])  
            out_hsi.append(hsi*self.gamma_hsi[i])
            out_rgb.append(rgb*self.gamma_rgb[i])
                                   
        hsi = self.bottleneck(torch.cat(out_hsi, 1), skipHSI, torch.cat(out_rgb, 1), skipRGB)
        
        del out_hsi, out_rgb
        
        if index > 0:
            hsi = torch.cat([self.gamma_feaContext[0]*hsi, self.gamma_feaContext[1]*feats_context], 1)
            hsi = self.feaFusion(hsi)
        
        feats_context = hsi        
        hsi = hsi + skipRGB 
       
        hsi = self.hsi_rgb_end(hsi)
        
        return hsi, feats_context
                                                             
        
                                             
