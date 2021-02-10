from torch.utils.data import DataLoader, random_split
import torch_optimizer as optim
from net import Transferrer, decoder, vgg_enc
from util import *
from tqdm import tqdm

def loaders(dataset_path, val_frac=.2, batch_size=8, image_size=512, doses=dose2locs.keys()):
    dataset = ImageDataset(dataset_path, image_size=image_size, doses=doses)
    val_len = int(len(dataset) * val_frac)
    lengths = [len(dataset) - val_len, val_len]
    datasets = random_split(dataset)
    return DataLoader(dataset[0], batch_size=batch_size*2), \
           DataLoader(dataset[1], batch_size=batch_size*2)

def reshape_batch(batch):
    n = loaders[0].batch_size/2
    return batch[:n], batch[n:]

def train(loaders, vgg_path=None, epochs=1,
          decoder=decoder, alpha=1.0):

    opt = optim.RAdam(decoder.parameters())
    model = Transferrer(vgg_enc(vgg_path), decoder, alpha=alpha)
    
    for epoch_num in range(epochs):
        model.train()
        for batch in loaders[0]:
            loss_c, loss_s = model(reshape_batch(batch))
            print(loss_c, loss_s)
  
    
