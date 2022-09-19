# encoding:utf-8

import torch
import torch.nn.functional as F
import pdb
from PIL import Image
from torchvision import transforms
import numpy as np

def calc_padding(x_shape, patchsize, stride, padding=None):
    if padding is None:
        xdim = x_shape
        padvert = -(xdim[0] - patchsize) % stride
        padhorz = -(xdim[1] - patchsize) % stride

        padtop = int(np.floor(padvert / 2.0))
        padbottom = int(np.ceil(padvert / 2.0))
        padleft = int(np.floor(padhorz / 2.0))
        padright = int(np.ceil(padhorz / 2.0))
    else:
        padtop = padbottom = padleft = padright = padding

    return padtop, padbottom, padleft, padright
    

def patchIndex(img, stride, patchsize):
	
    padtop, padbottom, padleft, padright = calc_padding(img.shape[2:], patchsize, stride, padding=None) 

    xpad = F.pad(img, pad=(padleft, padright, padtop, padbottom))     
        
    B, C, W, H = xpad.shape

    Z = F.unfold(xpad, kernel_size=patchsize, stride=stride)
    B, C_kh_kw, L = Z.size()
    Z = Z.permute(0, 2, 1)
    Z = Z.view(B, L, -1, patchsize, patchsize)   


    row = int(W / stride) 
    col = int(H / stride) 
    

    index = []
    values = []    
    
    patch_index = []
    cos_values = []
        
    
    for i in range(row):

        for j in range(col):
            z = col*i + j
            patch = Z[:,z,:,:,:]  
            	       	  	
            if i == 0:  ## fisrt line    	
                if j == 0:   
                    patch_index.append(z+1)  
                    patch_index.append(z+col)
                    patch_index.append(z+col+1) 
                    
                    flag = 3
                                     	             	
                elif j == col-1:  
                    patch_index.append(z-1) 
                    patch_index.append(z+col-1) 
                    patch_index.append(z+col) 
           	
                    flag = 3
           	        
                else:  
                    patch_index.append(z-1) 
                    patch_index.append(z+1) 
                    patch_index.append(z+col-1) 
                    patch_index.append(z+col) 
                    patch_index.append(z+col+1)                     
                    flag = 5
  	
            elif i == row-1:  # last line
                if j == 0: 
                    patch_index.append(z-col) 
                    patch_index.append(z-col+1) 
                    patch_index.append(z+1) 
                    
                    flag = 3                	
                	
                elif j == col-1: 
                    patch_index.append(z-col-1)
                    patch_index.append(z-col)
                    patch_index.append(z-1)
                    
                    flag = 3
                                    
                else:
                    patch_index.append(z-col-1) 
                    patch_index.append(z-col) 
                    patch_index.append(z-col+1) 
                    patch_index.append(z-1) 
                    patch_index.append(z+1)
                    
                    flag = 5
                                    		
            else: 	## other line
            	
                if j == 0:           		
                    patch_index.append(z-col) 
                    patch_index.append(z-col+1) 
                    patch_index.append(z+col) 
                    patch_index.append(z+col+1) 
                    patch_index.append(z+1) 
                    
                    flag = 5           	
            		
                elif j == col-1: 
                    patch_index.append(z-col-1) 
                    patch_index.append(z-col) 
                    patch_index.append(z-1) 
                    patch_index.append(z+col-1) 
                    patch_index.append(z+col) 
                    
                    flag =5             	  	
            		
                else: 
            		
                    patch_index.append(z-col-1)
                    patch_index.append(z-col)
                    patch_index.append(z-col+1)
                    patch_index.append(z-1)
                    patch_index.append(z+1)
                    patch_index.append(z+col-1)
                    patch_index.append(z+col)
                    patch_index.append(z+col+1) 
                    
                    flag = 8
                
                
            patch = patch.view(B, -1)
                
            if flag == 3:                 	

                for k in range(3):                 
                    cos_values.append(F.cosine_similarity(patch, Z[:,patch_index[k],:,:,:].contiguous().view(B,-1)))   
    
                sort_values, sort_index = torch.topk(torch.cat([cos_values[0].unsqueeze(1), cos_values[1].unsqueeze(1), 
                                          cos_values[2].unsqueeze(1)], 1), 
                                          3, dim=1)
                
                for k in range(3):
                    sort_index[torch.eq(sort_index, k)] = patch_index[k] + 10	
                
                sort_index = sort_index - (torch.ones(sort_index.shape)*10).cuda()
                   
                index.append(sort_index.long())
                values.append(sort_values) 

                                                        	
            if flag == 5:

                for k in range(5):                 
                    cos_values.append(F.cosine_similarity(patch, Z[:,patch_index[k],:,:,:].contiguous().view(B,-1)))   
                    
                sort_values, sort_index = torch.topk(torch.cat([cos_values[0].unsqueeze(1), cos_values[1].unsqueeze(1), 
                                          cos_values[2].unsqueeze(1), cos_values[3].unsqueeze(1), cos_values[4].unsqueeze(1)], 1), 
                                          3, dim=1)
                    
                for k in range(5):
                    sort_index[torch.eq(sort_index, k)] = patch_index[k]+10
                sort_index = sort_index - (torch.ones(sort_index.shape)*10).cuda()

                index.append(sort_index.long())
                values.append(sort_values)   
                  	
            if flag == 8:

                for k in range(8):                 
                    cos_values.append(F.cosine_similarity(patch, Z[:,patch_index[k],:,:,:].contiguous().view(B,-1)))   
                    
                sort_values, sort_index = torch.topk(torch.cat([cos_values[0].unsqueeze(1), cos_values[1].unsqueeze(1), 
                                          cos_values[2].unsqueeze(1), cos_values[3].unsqueeze(1), cos_values[4].unsqueeze(1),
                                          cos_values[5].unsqueeze(1), cos_values[5].unsqueeze(1), cos_values[7].unsqueeze(1)], 1),
                                          3, dim=1)
                    
                for k in range(8):
                    sort_index[torch.eq(sort_index, k)] = patch_index[k]+10

                sort_index = sort_index - (torch.ones(sort_index.shape)*10).cuda()
                                    
                index.append(sort_index.long())
                values.append(sort_values)       
                      
            patch_index = []
            cos_values = []
            

    return  index, values #[f_index, s_index, t_index], [f_values, s_values, t_values]                	          
  	            
                    
if __name__ == "__main__":
    
    im = Image.open('2.jpg')
    im1 = Image.open('1.jpg')
    im2 = Image.open('3.jpg')

    toTensor = transforms.ToTensor()      
    im = toTensor(im)[:,0:96,0:96].unsqueeze(0) 
    im1 = toTensor(im1)[:,0:96,0:96].unsqueeze(0) 
    im2 = toTensor(im2)[:,0:96,0:96].unsqueeze(0) 
    
    img = torch.cat([im, im1, im2], 0)
    first_index, second_index, third_index = patchIndex(img)  
    pdb.set_trace()
    zz = 0                    
                    
