from statistics import mean

import torch
import torch_optimizer as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from net import Transferrer, decoder, vgg_enc
from util import *
from data import ImageDataset


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

def train(loaders, transferrer, epochs=1, device=None,
          alpha=1.0):
    device = device or torch.device("cuda:0")
    opt = optim.RAdam(transferrer.decoder.parameters())
    model = nn.DataParallel(transferrer)
    
    for epoch_num in range(epochs):
        model.train()
        pbar = tqdm(loaders[0])
        for batch in pbar:
            batch = batch.to(device)
            loss_c, loss_s, loss_r = step(model, batch, opt)
            pbar.set_description(f"Loss c: {loss_c.item():.3f}, s: {loss_s.item():.3f}, r: {loss_r.item():.3f}")

        opt.zero_grad()
        model.eval()
        all_losses = [[], [], []]
        pbar = tqdm(loaders[1])
        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(device)
                batch_losses = model(*reshape_batch(batch))
                loss_c, loss_s, loss_r = map(lambda x: x.mean(), batch_losses)
                pbar.set_description(f"Loss c: {loss_c:.3f}, s: {loss_s:.3f}, r: {loss_r:.3f}")
                list(map(lambda x, y: x.append(y.mean().item()), all_losses, batch_losses))

        print(list(map(mean, all_losses)))
    return decoder

def step(model, batch, opt):
    opt.zero_grad()
    batch_losses = model(*reshape_batch(batch))
    loss_c, loss_s, loss_r = map(lambda x: x.mean(), batch_losses)
    (loss_c + loss_s + loss_r).backward()
    for param in model.module.decoder.parameters():
        param.grad.data.clamp_(-1,1)
    opt.step()
    return loss_c, loss_s, loss_r
