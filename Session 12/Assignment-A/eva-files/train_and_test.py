
class Train_test:
  
  def __init__(self):
    self.train_losses = []
    self.test_losses = []
    self.train_acc = []
    self.test_acc = []
    self.train_acc_epoch_end = []
    self.train_loss_epoch_end = []

  def train_and_test(self,model, device, train_loader,test_loader, optimizer, criterion, epoch):
      
    
    self.train_acc,self.train_acc_epoch_end,self.train_losses,self.train_loss_epoch_end = self.train(model, device, train_loader, optimizer, criterion)
    self.test_losses,self.test_acc = self.test(model, device, criterion, test_loader)
    return self.train_acc,self.train_acc_epoch_end,self.train_losses,self.train_loss_epoch_end,self.test_losses,self.test_acc


  def train(self,model, device, train_loader, optimizer, criterion):

    import torch
    #Training & Testing Loops
    from tqdm import tqdm

    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
      # get samples
      data, target = data.to(device), target.to(device)

      # Init
      optimizer.zero_grad()
      # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
      # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

      # Predict
      y_pred = model(data)

      # Calculate loss
      #loss = F.nll_loss(y_pred, target)
      loss = criterion(y_pred, target)
      self.train_losses.append(loss)

      # Backpropagation
      loss.backward()
      optimizer.step()

      # Update pbar-tqdm
      
      pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()
      processed += len(data)

      pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
      self.train_acc.append(100*correct/processed)
    self.train_acc_epoch_end.append(self.train_acc[-1])
    self.train_loss_epoch_end.append(self.train_losses[-1])
    return self.train_acc,self.train_acc_epoch_end,self.train_losses,self.train_loss_epoch_end


  def test(self,model, device, criterion, test_loader):

      import torch
    #Training & Testing Loops
      from tqdm import tqdm

      model.eval()
      test_loss = 0
      correct = 0
      with torch.no_grad():
          for data, target in test_loader:
              data, target = data.to(device), target.to(device)
              output = model(data)
              test_loss += criterion(output, target).item()  # sum up batch loss
              pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
              correct += pred.eq(target.view_as(pred)).sum().item()
              

      test_loss /= len(test_loader.dataset)
      self.test_losses.append(test_loss)

      print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
          test_loss, correct, len(test_loader.dataset),
          100. * correct / len(test_loader.dataset)))
      
      self.test_acc.append(100. * correct / len(test_loader.dataset))
      return self.test_losses,self.test_acc


def one_cycle_lr_range_test(min_lr,max_lr,steps,initial_model,device,trainloader,testloader,criterion,optimizer):

  trainAcc = []
  testAcc = []
  trainLoss = []
  testLoss = []
  lrs = []
  model = initial_model
  criterion = criterion
  optimizer = optimizer
  increment = (max_lr-min_lr)/steps
  last_lr = min_lr

  for i in range(steps):
    epoch=0
    # print(i,min_lr,max_lr,increment,last_lr)
    optimizer.param_groups[0]['lr'] = last_lr
    lrs.append(optimizer.param_groups[0]['lr'])
    print(i,')','LR:',optimizer.param_groups[0]['lr'])

    train_test = Train_test()
    train_acc,train_acc_epoch_end,train_losses,test_losses,test_acc = train_test.train_and_test(model, device, trainloader,testloader, optimizer, criterion, epoch)
    
    trainAcc.append(train_acc_epoch_end[0])
    testAcc.append(test_acc[0])
    trainLoss.append(train_losses[0])
    testLoss.append(test_losses[0])
    
    model = initial_model
    criterion = criterion
    optimizer = optimizer
    last_lr = last_lr + increment

  return lrs,trainAcc,trainLoss,testAcc,testLoss