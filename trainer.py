from statistics import mean

import torch
import torch_optimizer as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from net import Transferrer, decoder, vgg_enc
from util import *


def loaders(dataset_path, val_frac=.2, batch_size=8, image_size=512, doses=dose2locs.keys()):
    dataset = ImageDataset(dataset_path, image_size=image_size, doses=doses)
    val_len = int(len(dataset) * val_frac)
    lengths = [len(dataset) - val_len, val_len]
    datasets = random_split(dataset, lengths=lengths)
    return DataLoader(datasets[0], batch_size=batch_size*2), \
           DataLoader(datasets[1], batch_size=batch_size*2)

def reshape_batch(batch):
    n = int(batch.shape[0]/2)
    return batch[:n], batch[n:2*n]  # in case of odd length!

def train(loaders, vgg_enc, epochs=1, device=None,
          decoder=decoder, alpha=1.0):
    device = device or torch.device("cuda:0")
    opt = optim.RAdam(decoder.parameters())
    model = Transferrer(vgg_enc, decoder, alpha=alpha).to(device)
    
    for epoch_num in range(epochs):
        model.train()
        pbar = tqdm(loaders[0])
        for batch in pbar:
            batch = batch.to(device)
            opt.zero_grad()
            loss_c, loss_s = model(*reshape_batch(batch))
            pbar.set_description(f"Loss c: {loss_c:.3f}, s: {loss_s:.3f}")
            (loss_c + loss_s).backward()
            opt.step()

        opt.zero_grad()
        model.eval()
        losses_c, losses_s = [], [] 
        pbar = tqdm(loaders[1])
        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(device)
                loss_c, loss_s = model(*reshape_batch(batch))
                pbar.set_description(f"Loss c: {loss_c:.3f}, s: {loss_s:.3f}")
                losses_c.append(loss_c.item())
                losses_s.append(loss_s.item())
        print(mean(losses_c), mean(losses_s))
    return decoder
