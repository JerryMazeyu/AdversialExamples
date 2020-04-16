import os
project_index = os.getcwd().find('AdversialExamples')
root = os.getcwd()[0:project_index] + 'AdversialExamples'
import sys
sys.path.append(root)
import torch
import torch.nn.functional as F
from model.CNNModel import CNN_model
from tartgetdata.GetData import dataloaders
from utils.vis import batch_show




class Attack(object):
    def __init__(self, target_net, attack_type='FGSM', criterion_func=F.cross_entropy):
        """
        攻击目标网络生成样本
        :param target_net: 攻击的目标网络，需要有attribute name
        :param attack_type:
        :param criterion_func:
        """
        self.target_net = target_net
        self.attack_type = attack_type
        self.creterion_func = criterion_func

    def fgsm_attack(self, x, hx, ifdodge=False, epsilon=0.5, x_min=-1, x_max=1):
        assert self.target_net.name == 'resnet18', ValueError("This model isn't supported yet!")
        x = x.clone().detach().requires_grad_(True)
        output = self.target_net(x)
        loss = -self.creterion_func(output, hx) if ifdodge else self.creterion_func(output, hx)
        self.target_net.zero_grad()
        loss.backward()
        pertubation = epsilon * x.grad.sign_()
        return torch.clamp(x + pertubation, x_min, x_max)

    def ifgsm_attack(self, x, hx, ifdodge=False, epsilon=0.5, ):
        pass




if __name__ == '__main__':
    at = Attack(CNN_model)
    inp, label = next(iter(dataloaders['train']))
    res = at.attack(inp, label)
    batch_show(res.detach(), label)
