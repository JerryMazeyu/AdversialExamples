import matplotlib.pyplot as plt
import torch
from utils.write_log import log_here
import os
project_index = os.getcwd().find('AdversialExamples')
root = os.getcwd()[0:project_index] + 'AdversialExamples'
import sys
sys.path.append(root)
from tartgetdata.GetData import dataloaders, class_names
import torchvision
import cv2
import numpy as np

def imshow(inp, ans=None, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    if ans == True:
        inp = cv2.copyMakeBorder(inp,1,1,1,1, cv2.BORDER_CONSTANT,value=(0, 255, 0))
    elif ans == False:
        inp = cv2.copyMakeBorder(inp,1,1,1,1, cv2.BORDER_CONSTANT,value=(255, 0, 0))
    else:
        pass
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def batch_show_with_ans(tensor, ans, classes=None):
    """
    修改自torch.utils.make_grid，增加带有label边框的一组batch图像
    :param tensor: 一个batch的图像 example-->  torch.randn(16, 3, 28, 28)
    :param ans: 列表，长度和batch_size相同，元素是boolean example--> [True, False, False, ...]
    :param classes: label 一个长度为batch_size的Tensor example--> tensor([1, 2, 5, ...])
    :return: True是绿色边框，False是红色边框的组合图
    """
    try:
        classes = list(classes.numpy())
    except:
        classes = ans
    if not torch.is_tensor(tensor):
        raise TypeError("Input should be a tensor.")

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    tensor_tuple = tensor.split(1, 0)
    num_pic = len(tensor_tuple)
    row_num = int(np.ceil(np.sqrt(num_pic)))
    col_num = int(np.floor(np.sqrt(num_pic)))
    fig, ax = plt.subplots(row_num, col_num)
    ax = ax.flatten()
    for i in range(num_pic):
        inp = tensor_tuple[i].squeeze()
        inp = inp * 255
        inp = inp.numpy().transpose((1, 2, 0)).astype(int)
        if ans[i]:
            inp = cv2.copyMakeBorder(inp, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 255, 0))
        else:
            inp = cv2.copyMakeBorder(inp, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(255, 0, 0))
        ax[i].set_title(classes[i])
        ax[i].imshow(inp)
        ax[i].axis('off')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
    plt.show()


def batch_show(inputs, classes, ans=True):
    """Show me a batch."""
    out = torchvision.utils.make_grid(inputs, padding=10, pad_value=0)
    imshow(out, ans=ans, title=[class_names[x] for x in classes])




if __name__ == '__main__':
    inputs, classes = next(iter(dataloaders['train']))
    ans = [True]*15 + [False]
    batch_show_with_ans(inputs, ans, classes)

