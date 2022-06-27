# coding:utf-8
import torch
import numpy as np
import scipy.io as sio
from torch.autograd import Variable
import os
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from data_utils import  ValsetFromFolder

from noDU_option import  opt
from fine_model import FineNet
from noDU_model import CoarseNet
from os import listdir
import scipy.io as scio
import pdb
from eval import *
from patch import *

def generate_coarse(coarse_model, lr_hsi, hr_rgb):

    data = np.zeros((hr_rgb.shape[2], hr_rgb.shape[3], lr_hsi.shape[1])).astype(np.float32) 
    index, values = patchIndex(hr_rgb.cuda(), opt.upscale_factor*opt.stride, opt.upscale_factor*opt.patchsize)   
     
    with torch.no_grad():
 
        SR = None
        feats_context = None

        if opt.cuda:
            lr_hsi = lr_hsi.cuda()
            hr_rgb = hr_rgb.cuda() 

        for i in range(lr_hsi.shape[1]):
            neigbor = []       	    
            if i==0:                                
                neigbor.append(lr_hsi[:,1,:,:].data.unsqueeze(1))
                neigbor.append(lr_hsi[:,2,:,:].data.unsqueeze(1))
                	                 		                	                
            elif i==lr_hsi.shape[1]-1:
                neigbor.append(lr_hsi[:,i-1,:,:].data.unsqueeze(1))
                neigbor.append(lr_hsi[:,i-2,:,:].data.unsqueeze(1))               	
            else:
                neigbor.append(lr_hsi[:,i-1,:,:].data.unsqueeze(1))
                neigbor.append(lr_hsi[:,i+1,:,:].data.unsqueeze(1))
                	
            single = lr_hsi[:,i,:,:]
            neigbor =  Variable(torch.cat(neigbor, 1))
             
            SR, feats_context = coarse_model(i, single, neigbor, hr_rgb, [index, values], SR, feats_context)   
            SR[SR>1] = 1
            SR[SR<0] = 0                
                       
            data[:,:,i] = SR.detach().cpu().numpy()
            	
            	
    return data        	

def generate_fine(c_result, hsi, hr_rgb, lr_hsi):
	

    fine_model = FineNet(opt)
    
    if opt.cuda:
        fine_model = nn.DataParallel(fine_model).cuda()
        L1Loss = nn.L1Loss().cuda()
        lr_hsi = lr_hsi.cuda()
        hr_rgb = hr_rgb.cuda()
        
    else:
        fine_model = fine_model.cpu()   
        L1Loss = nn.L1Loss()
   
    print('# parameters:', sum(param.numel() for param in fine_model.parameters())) 

    optimizer = torch.optim.Adam([{'params':fine_model.parameters(),'initial_lr':5e-5}], lr=5e-5, weight_decay=0, betas=(0.9, 0.999), eps=1e-08)

    for i in range(400):

        out, D_HSI = fine_model(torch.from_numpy(c_result.transpose(2,0,1)).unsqueeze(0).cuda(), hr_rgb, lr_hsi)
        
        D_MSI = torch.reshape(torch.matmul(P, torch.reshape(out, (out.shape[0], out.shape[1], out.shape[2] * out.shape[3]))),(out.shape[0], hr_rgb.shape[1], hr_rgb.shape[2], hr_rgb.shape[3]))
             
        #Unsupervised loss function
  
        Loss = 0.75*L1Loss(D_HSI, lr_hsi) + L1Loss(D_MSI, hr_rgb) #+ 0.01*SAM(D_HSI[0,].cpu().detach().numpy(),lr_hsi[0,].cpu().detach().numpy())

        optimizer.zero_grad()
        Loss.backward()
        
        nn.utils.clip_grad_norm_(fine_model.parameters(),0.4)

        optimizer.step()
           
        if i % 50 == 0:
            out = torch.squeeze(out).detach().cpu().numpy() 
            out = out.transpose(1,2,0)           
            psnr = PSNR(hsi, out)
            sam = SAM(hsi, out)
            ssim = SSIM(hsi, out)
            print('At the {0}th epoch the Loss,PSNR,SAM,SSIM are {1:.8f}, {2:.2f}, {3:.2f}, {4:.4f}.'.format(i, Loss, psnr, sam, ssim))

    out = torch.squeeze(out).detach().cpu().numpy()
    out = out.transpose(1,2,0)
            
    return out
    
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])
    
if __name__ == "__main__":

    PSNRs = []
    SSIMs = []
    SAMs = []  

    cPSNRs = []
    cSSIMs = []
    cSAMs = [] 
        
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    input_path = '/media/hdisk/liqiang/source/CAVE/test_mat/8/downsample/direction/'
    names = [x for x in listdir(input_path) if is_image_file(x)]    
    
    coarse_model = CoarseNet(opt)      

    if opt.cuda:
        coarse_model = nn.DataParallel(coarse_model).cuda()
    else:
        coarse_model = coarse_model.cpu()   
    print('# parameters:', sum(param.numel() for param in coarse_model.parameters())) 

    # Buliding coarse_model     
    model_path = 'result/CAVE_model_8_Kernel_2.pth'
    #'result/CAVE_model_' + str(opt.upscale_factor) + '_' + opt.interpolation + '_' + str(opt.n_module)+ '.pth'  
    print("=> loading premodel")  

    checkpoint = torch.load(model_path)          
    coarse_model.load_state_dict(checkpoint['model'])
 
    CSR = sio.loadmat('P_N_V2.mat')['P']
    P = Variable(torch.unsqueeze(torch.from_numpy(CSR), 0)).type(torch.cuda.FloatTensor) 
             
    for k in range(len(names)):
        mat = scio.loadmat(input_path + names[k])   
        lr_hsi = mat['LR'].astype(np.float32).transpose(2,0,1)
        hsi = mat['HR'].astype(np.float32)  
        hr_rgb = mat['RGB'].astype(np.float32).transpose(2,0,1)
            
        lr_hsi = Variable(torch.from_numpy(lr_hsi).unsqueeze(0))         
        hr_rgb = Variable(torch.from_numpy(hr_rgb).unsqueeze(0))      

        c_result = generate_coarse(coarse_model, lr_hsi, hr_rgb)
        
        c_result[c_result>1.]=1
        c_result[c_result<0]=0
        

        m_psnr = PSNR(hsi, c_result)
        m_sam = SAM(hsi, c_result)
        m_ssim = SSIM(hsi, c_result)
        cPSNRs.append(m_psnr)
        cSSIMs.append(m_ssim)  
        cSAMs.append(m_sam)     
        print("===The {}-th picture=====PSNR:{:.4f}=====SSIM:{:.5f}=====SAM:{:.4f}".format(k+1,  m_psnr, m_ssim, m_sam))  

#        f_result = generate_fine(c_result, hsi, hr_rgb, lr_hsi)
#        
#        f_result[f_result>1.]=1
#        f_result[f_result<0]=0    
#
#        m_psnr = PSNR(hsi, f_result)
#        m_sam = SAM(hsi, f_result)
#        m_ssim = SSIM(hsi, f_result)
#            
#        PSNRs.append(m_psnr)
#        SSIMs.append(m_ssim)
#        SAMs.append(m_sam)
#                        
#        print("===The {}-th picture=====PSNR:{:.4f}=====SSIM:{:.5f}=====SAM:{:.4f}".format(k+1,  m_psnr, m_ssim, m_sam))

    print("=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(np.mean(cPSNRs), np.mean(cSSIMs), np.mean(cSAMs)))     	
#    print("=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(np.mean(PSNRs), np.mean(SSIMs), np.mean(SAMs))) 
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                       