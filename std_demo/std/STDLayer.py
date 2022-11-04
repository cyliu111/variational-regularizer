"""
Implementation of a learnable Soft Threshhold Dynamics (STDLayer) Layer 
Jun Liu. 01/2020
"""
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch


class STDLayer(nn.Module):

    def __init__(
        self,
        nb_classes,
        nb_iterations=10, 
        nb_kerhalfsize=3,
    ):
        """
        :param nb_classes: number of classes
        :param nb_iterations: iterations number
        :param nb_kerhalfsize: the half size of neigbourhood
        """
        super(STDLayer, self).__init__()
        
        self.nb_iterations = nb_iterations
        self.nb_classes = nb_classes
        self.ker_halfsize=nb_kerhalfsize

		#Learnable version: sigma of Gasussian function; entropic parameter epsilon; regularization parameter lam
        self.nb_sigma = nn.Parameter(torch.FloatTensor([10.0]*nb_classes).view(nb_classes,1,1))
        self.entropy_epsilon=nn.Parameter(torch.FloatTensor([1.0]))
        self.lam=nn.Parameter(torch.FloatTensor([5.0]))

		#Fixed parmaters.
        #self.nb_sigma = Variable(torch.FloatTensor([10.0]*nb_classes).view(nb_classes,1,1),requires_grad=False).cuda()
        #self.entropy_epsilon=Variable(torch.FloatTensor([1.0]),requires_grad=False).cuda()
        #self.lam=Variable(torch.FloatTensor([5.0]),requires_grad=False).cuda()
        
        # softmax
        self.softmax = nn.Softmax2d()


    def forward(self,o):
        u = self.softmax(o*(self.entropy_epsilon**2.0)) 
        # std kernel    
        ker= STDLayer.STD_Kernel(self.nb_sigma,self.ker_halfsize)
        #main iteration
        for i in range(self.nb_iterations):
            #1. subgradient 
            q = F.conv2d(1.0-2.0*u, ker, padding=self.ker_halfsize, groups=self.nb_classes)
            #2. Softmax
            u=self.softmax((o-self.lam*q)*(self.entropy_epsilon**2.0))
        
        return u

    def STD_Kernel(sigma,halfsize):
        x,y=torch.meshgrid(torch.arange(-halfsize,halfsize+1),torch.arange(-halfsize,halfsize+1))
        x=x.cuda()
        y=y.cuda()
        ker=torch.exp(-(x.float()**2+y.float()**2)/(2.0*sigma**2))
        ker=ker/(ker.sum(-1,keepdim=True).sum(-2,keepdim=True)+1e-15)
        ker=ker.unsqueeze(1)
        return ker
