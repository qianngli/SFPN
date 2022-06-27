from os import listdir
from os.path import join
from torch.utils.data import Dataset
from newTransform import compose, normalize, toTensor, randomCrop, randomHorizontalFlip, randomVerticalFlip, randomRotation, Resize_train, Resize_test, randomRoll 

import scipy.io as scio
import numpy as np
import random  
import pdb
import torch 
from torch.autograd import Variable 
import os
from imresize import downscale, anisotropic_gaussian_kernel, isotropic_gaussian_kernel

def mean_std(dataset):

    if dataset == 'CAVE':
        hsi_mean = (0.09397, 0.0946, 0.08612, 0.08283, 0.0838, 0.07968, 0.07577, 0.07499, 0.07753, 0.07792, 0.08221, 0.08819, 0.09325, 0.09466, 0.09318, 0.09065, 0.0899, 0.09245, 0.09795, 0.1039, 0.1112, 0.1183, 0.1226, 0.1243, 0.1247, 0.1259, 0.1278, 0.1286, 0.1338, 0.1413, 0.1525)
        hsi_std = (0.02449, 0.04057, 0.04861, 0.05656, 0.05889, 0.05648, 0.05424, 0.05369, 0.05456, 0.04889, 0.04776, 0.04968, 0.05249, 0.05271, 0.05188, 0.05066, 0.05082, 0.05115, 0.05351, 0.05059, 0.04885, 0.0521, 0.05304, 0.05226, 0.0491, 0.04918, 0.05008, 0.04872, 0.05092, 0.05209, 0.05093)
        
        rgb_mean = (0.26953, 0.23581, 0.22852)
        rgb_std = (0.21943, 0.17884, 0.15847)
        
    elif dataset == 'Harvard':
        hsi_mean = (0.09397, 0.0946, 0.08612, 0.08283, 0.0838, 0.07968, 0.07577, 0.07499, 0.07753, 0.07792, 0.08221, 0.08819, 0.09325, 0.09466, 0.09318, 0.09065, 0.0899, 0.09245, 0.09795, 0.1039, 0.1112, 0.1183, 0.1226, 0.1243, 0.1247, 0.1259, 0.1278, 0.1286, 0.1338, 0.1413, 0.1525)
        hsi_std = (0.02449, 0.04057, 0.04861, 0.05656, 0.05889, 0.05648, 0.05424, 0.05369, 0.05456, 0.04889, 0.04776, 0.04968, 0.05249, 0.05271, 0.05188, 0.05066, 0.05082, 0.05115, 0.05351, 0.05059, 0.04885, 0.0521, 0.05304, 0.05226, 0.0491, 0.04918, 0.05008, 0.04872, 0.05092, 0.05209, 0.05093)
        
        rgb_mean = (0.26953, 0.23581, 0.22852)
        rgb_std = (0.21943, 0.17884, 0.15847)
            	
    elif dataset == 'ICVL':
        hsi_mean = (0.09397, 0.0946, 0.08612, 0.08283, 0.0838, 0.07968, 0.07577, 0.07499, 0.07753, 0.07792, 0.08221, 0.08819, 0.09325, 0.09466, 0.09318, 0.09065, 0.0899, 0.09245, 0.09795, 0.1039, 0.1112, 0.1183, 0.1226, 0.1243, 0.1247, 0.1259, 0.1278, 0.1286, 0.1338, 0.1413, 0.1525)
        hsi_std = (0.02449, 0.04057, 0.04861, 0.05656, 0.05889, 0.05648, 0.05424, 0.05369, 0.05456, 0.04889, 0.04776, 0.04968, 0.05249, 0.05271, 0.05188, 0.05066, 0.05082, 0.05115, 0.05351, 0.05059, 0.04885, 0.0521, 0.05304, 0.05226, 0.0491, 0.04918, 0.05008, 0.04872, 0.05092, 0.05209, 0.05093)
        
        rgb_mean = (0.26953, 0.23581, 0.22852)
        rgb_std = (0.21943, 0.17884, 0.15847)
            	
    else:  
        raise TypeError('There is no this dataset.')
	
    return hsi_mean, hsi_std, rgb_mean, rgb_std

    
def calculate_valid_crop_size(upscale_factor, patchSize):
    return  upscale_factor * patchSize
      
def train_hr_transform(crop_size):

    return compose([
      randomCrop(crop_size),
      randomHorizontalFlip(),
      randomVerticalFlip(),
      randomRoll(),
      randomRotation('90'),
      randomRotation('-90'),
      toTensor()     
    ])
 
def train_lr_transform(upscale_factor, interpolation):
    return compose([ 
        Resize_train(upscale_factor, interpolation)
    ])
        
class TrainsetFromFolder(Dataset):
    def __init__(self, dataset, dataset_dir,  upscale_factor, interpolation = 'Bicubic', patchSize = 28, crop_num = 48):
        super(TrainsetFromFolder, self).__init__()
        
        image_names = listdir(dataset_dir) 
        
        self.total_names = []
        for i in range(crop_num):
            for j in range(len(image_names)):              
                self.total_names.append(dataset_dir + image_names[j])
          
        random.shuffle(self.total_names)
                     
        crop_size = calculate_valid_crop_size(upscale_factor, patchSize)

        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(upscale_factor, interpolation)
        
        csr = scio.loadmat('P_N_V2.mat')['P'].astype(np.float32)
        self.csr = Variable(torch.from_numpy(csr))
        
    def __getitem__(self, index):
   
        mat = scio.loadmat(self.total_names[index], verify_compressed_data_integrity=False) 
        
        hsi = mat['hsi'].astype(np.float32)

        crop_hsi = self.hr_transform(hsi) 

        crop_hr_rgb = torch.matmul(self.csr, torch.reshape(crop_hsi, (crop_hsi.shape[0], crop_hsi.shape[1] * crop_hsi.shape[2])))
        crop_hr_rgb = torch.reshape(crop_hr_rgb, (3, crop_hsi.shape[1], crop_hsi.shape[2]))
                
        crop_lr_hsi = self.lr_transform(crop_hsi)      
          
        return crop_hsi, crop_lr_hsi, crop_hr_rgb

    def __len__(self):
        return len(self.total_names)
        
        
def test_lr_hsi_transform(upscale_factor, interpolation):
    return compose([ 
        Resize_test(upscale_factor, interpolation)
    ])
    
            
def test_hr_rgb_transform():
    return compose([ 
        toTensor()
    ])
    
def mrop(scale, width, height):
    W = width//scale
    H = height//scale
    
    return int(W*scale), int(H*scale)
    
class ValsetFromFolder(Dataset):
    def __init__(self, dataset, dataset_dir, upscale_factor, interpolation):
        super(ValsetFromFolder, self).__init__()
        
        image_names = listdir(dataset_dir) 

        self.total_names = []
        for j in range(len(image_names)):              
            self.total_names.append(dataset_dir + image_names[j])         
                     
        self.lr_hsi_transform = test_lr_hsi_transform(upscale_factor, interpolation)
        self.hr_rgb_transform = test_hr_rgb_transform()
        
        csr = scio.loadmat('P_N_V2.mat')['P'].astype(np.float32)
        self.csr = Variable(torch.from_numpy(csr))
        self.scale = upscale_factor
        self.names = image_names

    def __getitem__(self, index):
        mat = scio.loadmat(self.total_names[index], verify_compressed_data_integrity=False) 
        hsi = mat['hsi'].astype(np.float32).transpose(2,0,1)
        W, H = mrop(self.scale, hsi.shape[1], hsi.shape[2]) 
        
        hsi = torch.from_numpy(hsi)[:,:W,:H]

        hr_rgb = torch.matmul(self.csr, torch.reshape(hsi, (hsi.shape[0], hsi.shape[1] * hsi.shape[2])))
        hr_rgb = torch.reshape(hr_rgb, (3, hsi.shape[1], hsi.shape[2]))
                
        lr_hsi = self.lr_hsi_transform(hsi)  
        
#        kernel_type = 'isotropic'
#        
#        
#        out_path = '/media/hdisk/liqiang/source/CAVE/test_mat/' + str(self.scale) + '/isotropic_gaussian_kernel_4/'   
#        
        
             
   
#        kernel = scio.loadmat('f_set')['f_set'][0,0]
#        lr_hsi = downscale(hsi.numpy().transpose(1,2,0), kernel, np.array([self.scale, self.scale]), output_shape=None)   
#        lr_hsi = lr_hsi.astype(np.float32).transpose(2,0,1)  
#        lr_hsi = torch.from_numpy(lr_hsi)
        
        
#        out_path = '/media/hdisk/liqiang/source/CAVE/test_mat/' + str(self.scale) + '/isotropic_gaussian_kernel_4/'
#        if not os.path.exists(out_path):
#           os.makedirs(out_path) 
#
#        scale_factor = 8
#        k_size = 21
#        scale = np.array([scale_factor, scale_factor])  
#        kernel_size = np.array([k_size, k_size])

#        lambda_1 = 2
##        lambda_2 = 10
##        theta = 0.5 * np.pi                
##        kernel = anisotropic_gaussian_kernel(kernel_size, scale, lambda_1, lambda_2, theta)  
#             
#        sigma = 4     
#        kernel = isotropic_gaussian_kernel(kernel_size, scale, sigma)
#        
#        
#        lr_hsi = downscale(hsi.numpy().transpose(1,2,0), kernel, scale)     
#              
#        print(lr_hsi.numpy().transpose(1,2,0).shape)
#        print(hr_rgb.transpose(1,2,0).shape)        
#        print(hsi.numpy().transpose(1,2,0).shape)
#        
#        scio.savemat(out_path + self.names[index], {'LR': lr_hsi,'RGB': hr_rgb.numpy().transpose(1,2,0), 'HR': hsi.numpy().transpose(1,2,0)}) 
#        	               
        hr_rgb = self.hr_rgb_transform(hr_rgb)          
        
        return hsi, lr_hsi, hr_rgb, self.names[index]

    def __len__(self):
        return len(self.total_names)