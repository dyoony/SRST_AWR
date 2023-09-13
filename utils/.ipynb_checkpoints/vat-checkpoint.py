import os
import time
import math
import matplotlib as mpl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
    

###############################################################################
## VAT model

class VAT(nn.Module):
    def __init__(self  , batch_size , input_size , xi , epsilon):
        super(VAT, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.xi = xi
        self.epsilon = epsilon
        self.advr = nn.Parameter(torch.randn([self.batch_size]+self.input_size) , requires_grad=True)
                
    def normalize_advr(self):
        advr_val = self.advr.data
        advr_val /= (1e-12+(advr_val.abs().max()))
        norm_advr = advr_val.pow(2)

        for k in range(1,(len(self.input_size)+1)):
            norm_advr = norm_advr.sum(dim=k , keepdim=True)

        norm_advr = norm_advr.pow(0.5)
        advr_val /= norm_advr
        
        self.advr.data = advr_val
         
    def KL_divergence(self , x , out_x , feedforward_model):
        _ , out_xr = feedforward_model(x+self.advr)

        softmax_x = Variable(nn.Softmax(1)(out_x).data)        ## not learned, supervisor!
        logsoftmax_x = Variable(nn.LogSoftmax(1)(out_x).data)  ## not learned, supervisor!
        logsoftmax_xr = nn.LogSoftmax(1)(out_xr)
       
        return (softmax_x*(logsoftmax_x-logsoftmax_xr)).sum(1).mean()


###############################################################################
## VAT loss

def VAT_loss(input, model, vat, use_cuda):

    output = model(input)        
    
    #######################################
    ## calculate adversarial noise matrix
    ##############################
    ## for unlabeled data
    advr_matrix = torch.randn(input.size())
    if use_cuda: advr_matrix = advr_matrix.cuda()
        
    vat.advr.data = advr_matrix                          
    ## normalize advr to have norm xi
    vat.normalize_advr(); vat.advr.data *= vat.xi

    ## calculate KL divergence between standard and noised distribution
    kl_div = vat.KL_divergence(input , output , model)*1e+5  ## for numerical stability
    ## get gradient
    vat.zero_grad()            
    advr_grad = torch.autograd.grad(kl_div , vat.advr)[0].data
    ## insert the normalized gradient to advr
    vat.advr.data = advr_grad
    vat.normalize_advr(); vat.advr.data *= vat.epsilon
                            
    ## calculate kl_div
    vat_loss = vat.KL_divergence(input , output , model)
    
    return vat_loss


