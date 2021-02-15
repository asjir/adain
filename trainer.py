from statistics import mean

import torch
import torch_optimizer as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data import ImageDataset
from util import *


def loaders(dataset_path, val_frac=.2, batch_size=8, image_size=512, doses=dose2locs.keys(),
            aug_prob=0., norm_f=None):
    dataset = ImageDataset(dataset_path, image_size=image_size, doses=doses, aug_prob=aug_prob, norm_f=norm_f)
    val_len = int(len(dataset) * val_frac)
    lengths = [len(dataset) - val_len, val_len]
    datasets = random_split(dataset, lengths=lengths)
    return DataLoader(datasets[0], batch_size=batch_size*2), \
           DataLoader(datasets[1], batch_size=batch_size*2)

def reshape_batch(batch):
    # TODO: it needs to be put in dataset
    n = int(batch.shape[0]/2)
    return batch[:n], batch[n:2*n]  # in case of odd length!

def train(loaders, transferrer, epochs=1, device=None):
    device = device or torch.device("cuda:0")
    opt = optim.RAdam(transferrer.decoder.parameters())
    model = nn.DataParallel(transferrer).to(device)
    
    for epoch_num in range(epochs):
        model.train()
        pbar = tqdm(loaders[0])
        for batch_content, batch_style in pbar:
            batch_content.to(device); batch_style.to(device)
            loss_c, loss_s, loss_r = step(model, batch_content, batch_style, opt)
            pbar.set_description(f"Loss c: {loss_c.item():.3f}, s: {loss_s.item():.3f}, r: {loss_r.item():.3f}")

        opt.zero_grad()
        model.eval()
        all_losses = [[], [], []]
        pbar = tqdm(loaders[1])
        with torch.no_grad():
            for batch_content, batch_style in pbar:
                batch_content.to(device); batch_style.to(device)
                batch_losses = model(batch_content, batch_style)
                loss_c, loss_s, loss_r = map(lambda x: x.mean(), batch_losses)
                pbar.set_description(f"Loss c: {loss_c:.3f}, s: {loss_s:.3f}, r: {loss_r:.3f}")
                list(map(lambda x, y: x.append(y.mean().item()), all_losses, batch_losses))

        print(list(map(mean, all_losses)))
    return transferrer

def step(model, batch_content, batch_style, opt):
    opt.zero_grad()
    batch_losses = model(batch_content, batch_style)
    loss_c, loss_s, loss_r = map(lambda x: x.mean(), batch_losses)
    (loss_c + loss_s + loss_r).backward()
    for param in model.module.decoder.parameters():
        param.grad.data.clamp_(-1,1)
    opt.step()
    return loss_c, loss_s, loss_r
