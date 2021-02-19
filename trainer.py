from statistics import mean
from typing import Tuple

import torch
import torch_optimizer as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data import ImageDataset
from util import *


def homogenous_collate_fn(batch):
    n = int(len(batch)/2)
    b1 = batch[:n]
    b2 = batch[n:n*2]
    f = torch.utils.data.dataloader.default_collate
    return f(b1), f(b2)

def loaders(dataset_path, val_frac=.2, batch_size=8, image_size=512, doses=dose2locs.keys(),
            aug_prob=0., norm_f=None, num_workers=1):
    dataset = ImageDataset(dataset_path, image_size=image_size, doses=doses, aug_prob=aug_prob, norm_f=norm_f)
    val_len = int(len(dataset) * val_frac)
    lengths = [len(dataset) - val_len, val_len]
    datasets = random_split(dataset, lengths=lengths)
    f = lambda x: DataLoader(x, batch_size=batch_size*2, collate_fn=homogenous_collate_fn, num_workers=num_workers)
    return f(datasets[0]), f(datasets[1])


def reshape_batch(batch):
    # TODO: it needs to be put in dataset
    n = int(batch.shape[0]/2)
    return batch[:n], batch[n:2*n]  # in case of odd length!


def train(loaders:Tuple[DataLoader], transferrer:nn.Module, model_dir=None, epochs=1, device=None):
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
        
        print(evaluate(model, loaders[1]))
        
        if "step" in dir(loaders[0].dataset):
            loaders[0].dataset.step()
        if model_dir:
            torch.save(transferrer.state_dict(), Path(model_dir) / f"model@epoch{epoch_num}.pt")
        
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


def evaluate(model, eval_loader):
    model.eval()
    all_losses = [[], [], []]
    pbar = tqdm(eval_loader)
    with torch.no_grad():
        for batch_content, batch_style in pbar:
            batch_content.to(device); batch_style.to(device)
            batch_losses = model(batch_content, batch_style)
            loss_c, loss_s, loss_r = map(lambda x: x.mean(), batch_losses)
            pbar.set_description(f"Loss c: {loss_c:.3f}, s: {loss_s:.3f}, r: {loss_r:.3f}")
            list(map(lambda x, y: x.append(y.mean().item()), all_losses, batch_losses))
    return list(map(mean, all_losses))

