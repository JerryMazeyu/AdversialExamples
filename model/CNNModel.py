from torchvision.models import resnet18
import torch
from torch import nn
from torch import optim
import time
import copy
import os
project_index = os.getcwd().find('AdversialExamples')
root = os.getcwd()[0:project_index] + 'AdversialExamples'
import sys
sys.path.append(root)
from tartgetdata.GetData import dataloaders, dataset_sizes
from config.Config import opt
import os
from torch.optim import lr_scheduler
from utils.write_log import log_here




"""
用resnet18作为分类器，并训练, 将参数存在/checkpoints中
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CNN_model = resnet18(pretrained=True)   # 可以选择别的网络
CNN_model.name = 'resnet18'

for param in CNN_model.parameters():
    param.requires_grad = False

num_features = CNN_model.fc.in_features
CNN_model.fc = nn.Linear(num_features, 10)

model_ft = CNN_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(CNN_model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)



def train_model(model, criterion, optimizer, scheduler, num_epochs=25, save_path='checkpoints/'):
    since = time.time()
    path = root + '/checkpoints/model.pth'
    try:
        model.load_state_dict(torch.load(path))
        log_here('info', "load path OK!")
    except:
        log_here('debug', "no model.pth in %s" % path)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        epoch_info = 'Epoch {}/{}'.format(epoch, num_epochs - 1)
        blank = '-' * 10
        log_here('info', epoch_info)
        log_here('info', blank)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over tartgetdata.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            loss_info = '{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)
            log_here('info', loss_info)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                prefix = 'model' + str(epoch) + '.pth'
                torch.save(best_model_wts, os.path.join(save_path, prefix))
                prefix_1 = 'model.pth'
                torch.save(best_model_wts, os.path.join(save_path, prefix_1))
                log_here('info', "Model has been saved as %s!" % prefix)

    time_elapsed = time.time() - since
    time_info = 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
    val_info = 'Best val Acc: {:4f}'.format(best_acc)
    log_here('info', time_info)
    log_here('info', val_info)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



if __name__ == '__main__':
    CNN_model = train_model(CNN_model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=opt.epochs)



