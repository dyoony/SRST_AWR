import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def UAT_FT_loss(inputs_lb, targets_lb, inputs_ul, model, attack, teacher_model):
    if teacher_model is None:
        raise ValueError("teacher_model is None.")
    #inputs_ul = torch.cat([inputs_lb, inputs_ul_w])
    inputs_lb = torch.cat([inputs_lb, inputs_ul]) 
    targets_lb = torch.cat([targets_lb, teacher_model(inputs_ul).argmax(dim=1).detach()])
    
    adv_inputs_lb, _ = attack.perturb(inputs_lb, targets_lb)
    adv_outputs_lb = model(adv_inputs_lb)
    sup_loss = F.cross_entropy(adv_outputs_lb, targets_lb)
    #rob_loss = F.kl_div((adv_probs_ul_w+1e-12).log(), nat_probs_ul_w, reduction='none').sum(dim=1).mean()

    return sup_loss


def UAT_PP_loss(inputs_lb, targets_lb, inputs_ul, model, attack, teacher_model):
    if teacher_model is None:
        raise ValueError("teacher_model is None.")
    #inputs_ul = torch.cat([inputs_lb, inputs_ul_w])
    inputs_lb = torch.cat([inputs_lb, inputs_ul])
    targets_lb = torch.cat([targets_lb, teacher_model(inputs_ul).argmax(dim=1).detach()])
    
    adv_inputs_lb, _ = attack.perturb(inputs_lb, targets_lb)
    adv_outputs_lb = model(adv_inputs_lb)
    sup_loss = F.cross_entropy(adv_outputs_lb, targets_lb)
    
    reg_loss = F.kl_div(F.log_softmax(adv_outputs_lb, dim=1), F.softmax(model(inputs_lb), dim=1).detach(), reduction = 'batchmean')
    #rob_loss = F.kl_div((adv_probs_ul_w+1e-12).log(), nat_probs_ul_w, reduction='none').sum(dim=1).mean()

    return sup_loss, reg_loss


# labeled data : unlabeled data = 1: 1 => default in RST
def RST_loss(inputs_lb, targets_lb, inputs_ul, model, attack, teacher_model):
    if teacher_model is None:
        raise ValueError("teacher_model is None.")
        
    inputs_lb = torch.cat([inputs_lb, inputs_ul]) 
    
    targets_lb = torch.cat([targets_lb, teacher_model(inputs_ul).argmax(dim=1).detach()])
    outputs_lb = model(inputs_lb) 
    
    sup_loss = F.cross_entropy(outputs_lb, targets_lb)
    
    adv_inputs_lb, _ = attack.perturb(inputs_lb)
    adv_outputs_lb = model(adv_inputs_lb)
    
    reg_loss = F.kl_div(F.log_softmax(adv_outputs_lb, dim=1), F.softmax(outputs_lb, dim=1), reduction = 'batchmean')
    #rob_loss = F.kl_div((adv_probs_ul_w+1e-12).log(), nat_probs_ul_w, reduction='none').sum(dim=1).mean()

    return sup_loss, reg_loss


def TRADES_loss(inputs_lb, targets_lb, inputs_ul, model, attack):
    sup_loss = F.cross_entropy(model(inputs_lb), targets_lb)
    outputs_ul = model(inputs_ul)
    #adv_inputs_ul, _ = attack.perturb(inputs_ul)
    adv_inputs_ul, _ = attack.perturb(inputs_ul)
    adv_outputs_ul = model(adv_inputs_ul)
    adv_probs_ul = F.softmax(adv_outputs_ul, dim=1)
    nat_probs_ul = F.softmax(model(inputs_ul), dim=1)

    reg_loss = F.kl_div((adv_probs_ul+1e-12).log(), nat_probs_ul, reduction = 'batchmean')

    return sup_loss, reg_loss

def Semi_TRADES_loss(inputs_lb, targets_lb, inputs_ul, model, attack, tau, teacher_model=None):
    if teacher_model is not None:
        sup_loss = F.cross_entropy(model(inputs_lb), targets_lb)
        outputs_ul = model(inputs_ul)
        knowledge_loss = F.kl_div((F.softmax(outputs_ul/tau, dim=1) + 1e-12).log(), F.softmax(teacher_model(inputs_ul)/tau, dim=1) + 1e-12, reduction='batchmean')
        adv_inputs_ul, _ = attack.perturb(inputs_ul, teacher_model(inputs_ul).argmax(dim=1).detach())
        adv_outputs_ul = model(adv_inputs_ul)
        adv_probs_ul = F.softmax(adv_outputs_ul, dim=1)
        nat_probs_ul = F.softmax(model(inputs_ul), dim=1)
        
        reg_loss = F.kl_div((adv_probs_ul+1e-12).log(), nat_probs_ul, reduction = 'batchmean')
        #adv_probs_ul_w = F.softmax(adv_outputs_ul_w, dim=1)
        #nat_probs_ul_w = F.softmax(model(inputs_ul_w), dim=1)
        #rob_loss = F.kl_div((adv_probs_ul_w+1e-12).log(), nat_probs_ul_w, reduction='none').sum(dim=1).mean()
        #rob_loss = F.kl_div((adv_probs_ul_w+1e-12).log(), nat_probs_ul_w, reduction='none').sum(dim=1).mean()
        
        return sup_loss, reg_loss, knowledge_loss
        
    else:
        raise ValueError("teacher_model is None.")


        

def AWR_loss(inputs_lb, targets_lb, inputs_ul, model, attack, smoothing, beta, teacher_model=None):
    LS_loss = LabelSmoothingCrossEntropy(smoothing)
    inputs_lb = torch.cat([inputs_lb, inputs_ul]) 
    
    targets_lb = torch.cat([targets_lb, teacher_model(inputs_ul).argmax(dim=1).detach()])
    outputs_lb = model(inputs_lb) 
    
    sup_loss = LS_loss(outputs_lb, targets_lb)
    
    adv_inputs_lb, _ = attack.perturb(inputs_lb)
    adv_outputs_lb = model(adv_inputs_lb)
    adv_probs_lb = F.softmax(adv_outputs_lb, dim=1)
    nat_probs_lb = F.softmax(model(inputs_lb), dim=1)
    teacher_nat_probs_lb = F.softmax(teacher_model(inputs_lb), dim=1).detach()
    teacher_adv_probs_lb = F.softmax(teacher_model(adv_inputs_lb), dim=1).detach()
    
    conv_weight = (teacher_nat_probs_lb * nat_probs_lb).sum(dim=1).mul(beta) + (1. - (teacher_adv_probs_lb * adv_probs_lb).sum(dim=1)).mul(1-beta)
    
    reg_loss = (F.kl_div((adv_probs_lb+1e-12).log(), nat_probs_lb, reduction='none').sum(dim=1) * conv_weight).mean()

    return sup_loss, reg_loss

def SRST_AWR_KD_loss(inputs_lb, targets_lb, inputs_ul, model, attack, smoothing, beta, tau, teacher_model=None):
    LS_loss = LabelSmoothingCrossEntropy(smoothing)
    # pseudo labeling
    sup_loss = LS_loss(model(inputs_lb), targets_lb)
    #sup_loss = LS_loss(model(inputs_lb), targets_lb) + LS_loss(model(inputs_ul), teacher_model(inputs_ul).argmax(dim=1).detach())
    outputs_ul = model(inputs_ul)
    knowledge_loss = F.kl_div((F.softmax(outputs_ul/tau, dim=1) + 1e-12).log(), F.softmax(teacher_model(inputs_ul)/tau, dim=1) + 1e-12, reduction='batchmean')
    
    adv_inputs_ul, _ = attack.perturb(inputs_ul, teacher_model(inputs_ul).argmax(dim=1).detach())
    adv_outputs_ul = model(adv_inputs_ul)
    adv_probs_ul = F.softmax(adv_outputs_ul, dim=1)
    nat_probs_ul = F.softmax(model(inputs_ul), dim=1)
    
    teacher_nat_probs_ul = F.softmax(teacher_model(inputs_ul), dim=1).detach()
    teacher_adv_probs_ul = F.softmax(teacher_model(adv_inputs_ul), dim=1).detach()
    
    #print(teacher_nat_probs_ul.shape)
    conv_weight = (teacher_nat_probs_ul * nat_probs_ul).sum(dim=1).mul(beta) + (1. - (teacher_nat_probs_ul * adv_probs_ul).sum(dim=1)).mul(1-beta)
    #conv_weight = (teacher_nat_probs_ul * nat_probs_ul).sum(dim=1).div(2) + (1. - (teacher_nat_probs_ul * adv_probs_ul).sum(dim=1)).div(2)
    #print(conv_weight.shape) [128, 10]
    
    #nat_teacher_probs_ul = F.softmax(teacher_model(inputs_ul), dim=1)
    #adv_teacher_conf_ul = torch.gather(adv_probs_ul, 1, (nat_teacher_probs_ul.argmax(dim=1).unsqueeze(1)).long()).squeeze()

    #adv_conf_ul = torch.gather(adv_probs_ul, 1, (adv_probs_ul.argmax(dim=1).unsqueeze(1)).long()).squeeze()
    reg_loss = (F.kl_div((adv_probs_ul+1e-12).log(), nat_probs_ul, reduction='none').sum(dim=1) * conv_weight).mean()

    return sup_loss, reg_loss, knowledge_loss
        
    
    
def Fix_Match_loss(inputs_lb, targets_lb, inputs_ul_w, inputs_ul_s, model, eta):
    logits_lb = model(inputs_lb)
    logits_ul_w = model(inputs_ul_w)
    logits_ul_s = model(inputs_ul_s)
                                  
    batch_size = inputs_lb.shape[0]
    # (l, ul_w, ul_s) = (64, 128, 128)                                  
    inputs = interleave(
            torch.cat((inputs_lb, inputs_ul_w, inputs_ul_s)), 2*2+1)                                  
    logits = model(inputs)
    logits = de_interleave(logits, 2*2+1)
    logits_lb = logits[:batch_size]
    logits_ul_w, logits_ul_s = logits[batch_size:].chunk(2)
    del logits                              
                                  
    sup_loss = F.cross_entropy(logits_lb, targets_lb, reduction='mean')
    pred_prob = torch.softmax(logits_ul_w.detach(), dim=-1)
    max_probs, targets_ul = torch.max(pred_prob, dim=-1)
    mask = max_probs.ge(eta).float()

    con_loss = (F.cross_entropy(logits_ul_s, targets_ul,
                          reduction='none') * mask).mean()

        
    return sup_loss, con_loss

                                  
def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])



def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p*F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()