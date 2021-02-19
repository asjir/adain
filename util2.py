from util import * 
from net import Transferrer

def reconstruction_loss(transferer:Transferrer, eval_loader):
    transferer.consistency_loss = nn.L1Loss()
    