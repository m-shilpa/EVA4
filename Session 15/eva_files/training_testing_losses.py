import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from kornia import losses


class DiceLoss(nn.Module):

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps = 1e-6

    def forward(self,pred,target):

        predict = torch.sigmoid(pred)

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(predict * target, dims)
        cardinality = torch.sum(predict + target, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(torch.tensor(1.) - dice_score)

class Dice_Metric(nn.Module):

    def __init__(self) -> None:
        super(Dice_Metric, self).__init__()
        self.eps = 1e-6

    def forward(self,pred,target):

        predict = torch.sigmoid(pred)

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(predict * target, dims)
        cardinality = torch.sum(predict + target, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(dice_score)

class BCEDiceLoss(nn.Module):
    def __init__(self) -> None:
        super(BCEDiceLoss, self).__init__()
        self.dice_loss = DiceLoss()

    def forward(self,pred,target):

        bce_loss = F.binary_cross_entropy_with_logits(pred, target)
        dice_loss = self.dice_loss(pred, target)

        loss = bce_loss + 2*dice_loss

        return loss

class MSEDiceLoss(nn.Module):
    def __init__(self) -> None:
        super(MSEDiceLoss, self).__init__()
        self.dice_loss = DiceLoss()

    def forward(self,pred,target):

        mse_loss = F.mse_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)

        loss = (mse_loss + dice_loss)/2

        return loss

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,pred,y):
        return torch.sqrt(self.mse(pred,y))

class MSE_SSIMLoss(nn.Module):
    def __init__(self) -> None:
        super(MSE_SSIMLoss, self).__init__()
        self.ssim = losses.SSIM(window_size=3, reduction= 'mean')

    def forward(self,pred,target):
        pred = torch.sigmoid(pred)
        mse_loss = F.mse_loss(pred, target)
        ssim_loss = self.ssim(pred, target)

        loss = (mse_loss + torch.mean(ssim_loss))/2

        return loss

class BCE_SmoothL1Loss(nn.Module):
    def __init__(self) -> None:
        super(BCE_SmoothL1Loss, self).__init__()
        

    def forward(self,pred,target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target)
        smoothl1_loss = F.smooth_l1_loss(pred, target)

        loss = (bce_loss + smoothl1_loss)/2

        return loss

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

        self.ssim_loss_5x5 = losses.SSIM(5, reduction='none')
        self.ssim_loss_11x11 = losses.SSIM(11, reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_sig = torch.sigmoid(input)

        loss_5 = self.ssim_loss_5x5(input_sig, target)
        loss_11 = self.ssim_loss_11x11(input_sig, target)

        return torch.mean(loss_5) + torch.mean(loss_11)

def save_model(epoch , model, optimizer, stats ,best_loss1,path ):
  torch.save({
            'epoch': epoch,
            'stats':stats,
            'state_dict': model,
            'best_loss1': best_loss1,
            'optimizer' : optimizer,
        }, path)


def iou_cal(outputs,target):
    dims = (1, 2, 3)
    outputs = torch.sigmoid(outputs)
    intersection = torch.sum((outputs * target), dims)
    union = torch.sum((outputs + target), dims)
    iou = (intersection + 1e-6) /(union + 1e-6)
    return torch.mean(iou)

def Ssim_index(loss):
  return 1-(loss*2)

class Train_model():
  
  def __init__(self,stats,Epoch=0):
    self.stats = stats
    self.mask_ious = []
    self.ssim_indices = []
    self.mask_loss = []
    self.depth_loss = []
    self.train_loss = []
    self.mask_ious1 = []
    self.ssim_indices1 = []
    self.mask_loss1 = []
    self.depth_loss1 = []
    self.train_loss1 = []
    self.best_loss1 = 100000
    self.Epoch = Epoch
  

  # def train(self,model, device, train_loader,test_loader, optimizer, criterion, epoch):
      
    
  #   self.train_acc,self.train_acc_epoch_end,self.train_losses,self.train_loss_epoch_end = self.train(model, device, train_loader, optimizer, criterion)
    
  #   return self.train_acc,self.train_acc_epoch_end,self.train_losses,self.train_loss_epoch_end


  def train(self,model, device, train_loader, optimizer, criterion1,criterion2,epoch,path):

    import torch
    #Training & Testing Loops
    from tqdm import tqdm

    model.train()
    pbar = tqdm(train_loader)
    mask_coef = 0
    ssim_index = 0
    for batch_idx, (bg,fg_bg,mask,depth) in enumerate(pbar):

      data = torch.cat((fg_bg,bg),1) 
    
      data,mask,depth = data.to(device), mask.to(device), depth.to(device)

      optimizer.zero_grad()
      mask_pred, depth_pred = model(data)

      
      loss1 = criterion1(mask_pred, mask.unsqueeze(1))
      loss2 =  criterion2(depth_pred, depth.unsqueeze(1)) 
      loss = loss1 + loss2
      
      
      # Backpropagation
      loss.backward()
      optimizer.step()

      lr = optimizer.param_groups[0]['lr']
      pbar.set_description(desc= f'Epoch= {epoch+1} LR= {lr} Mask Loss={loss1.item()} Depth Loss={loss2.item()} Loss={loss.item()} Batch_id={batch_idx}')
      mask_coef += dice_coefficient(mask_pred,mask, mask= True).item()
      ssim_index +=  Ssim_index(loss2.item())

    self.mask_ious1.append(mask_coef)
    self.ssim_indices1.append(ssim_index)
    self.mask_loss1.append(loss1.item())
    self.depth_loss1.append(loss2.item())
    self.train_loss1.append(loss.item())

    self.mask_ious.append(mask_coef.item())
    self.ssim_indices.append(ssim_index)
    self.mask_loss.append(loss1.item())
    self.depth_loss.append(loss2.item())
    self.train_loss.append(loss.item())

    if (self.train_loss[-1] < self.best_loss1):
      self.best_loss1 = self.train_loss[-1]
      self.stats['mask_loss'].extend(self.mask_loss),
      self.stats['depth_loss'].extend(self.depth_loss),
      self.stats['train_loss'].extend(self.train_loss),
      self.stats['mask_iou'].extend(self.mask_ious),
      self.stats['ssim_index'].extend(self.ssim_indices),

      save_model(epoch = self.Epoch + 1, model = model.state_dict(), optimizer = optimizer.state_dict(), stats = self.stats,
                 best_loss1 =self.train_loss[-1],path= path)
    
      self.mask_ious = []
      self.ssim_indices = []
      self.mask_loss = []
      self.depth_loss = []
      self.train_loss = []
    self.Epoch += 1  


    return self.stats,mask,depth,mask_pred, depth_pred


def test(model, device, test_loader, optimizer, criterion1,criterion2):

    import torch
    #Training & Testing Loops
    from tqdm import tqdm
    mask_ious = []
    dice_met = []
    model.train()
    pbar = tqdm(test_loader)
    mask_coef = 0
    ssim_index  = 0
    for batch_idx, (bg,fg_bg,mask,depth) in enumerate(pbar):

      data = torch.cat((fg_bg,bg),1)      
      data,mask,depth = data.to(device), mask.to(device), depth.to(device)

      optimizer.zero_grad()
      mask_pred, depth_pred = model(data)

      
      loss1 = criterion1(mask_pred, mask.unsqueeze(1))
      loss2 =  criterion2(depth_pred, depth.unsqueeze(1)) 
      loss = (loss1 + loss2)/2
      # dice_met.append(met(mask_pred,mask).item())
      mask_coef += dice_coefficient(mask_pred,mask, mask= True).item()
      pbar.set_description(desc= f'Mask Loss={loss1.item()} Depth Loss={loss2.item()} Loss={loss.item()} Batch_id={batch_idx}')
      

    # mask_iou = iou_cal(mask_pred,mask)
    ssim_index +=  Ssim_index(loss2.item()) 
    # print(f'\n Mask IOU = {mask_coef}, SSIM Index = {ssim_index},  Mask Loss =  {loss1.item()}, Depth Loss = {loss2.item()}, Avg. Loss = {loss.item()}')
    return dice_met,mask_coef,ssim_index,loss1.item(),loss2.item(),loss.item(),mask,depth,mask_pred,depth_pred


import torch
from torch.autograd import Function


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coefficient(pred, target, mask=False):
    """Dice coeff for batches"""
    pred = torch.sigmoid(pred)
    s = torch.FloatTensor(1).cuda().zero_()

    for i, c in enumerate(zip(pred, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)
