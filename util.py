from pathlib import Path

import torch
from matplotlib import pyplot as plt
from more_itertools import take
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.utils import make_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dose2locs = {
    0.0: ['B10','C5','C9','C10','E2','E7','F7','G7'],
    0.1: ['C2','C11','D6','D8','D10','E4','E8','F3','G9'],
    0.3: ['B5', 'D7', 'D9','D11','E6','E11','F6','F10','F11'],
    1.0: ['B4','B7','C4','E5','E9','F4','F8','G5','G6'],
    3.0: ['B6','D2','D3','D4','E3','F9','G3','G4'],
    30.0: ['B8','B9','C3','C6','C7','C8','E10','F2','G8']
}
loc2dose = dict()
for k, vs in dose2locs.items():
    for v in vs:
        loc2dose[v] = k

def show(g):
    plt.figure(figsize=(40,40)) 
    plt.imshow(g[0,:3].detach().cpu().permute(1,2,0))

def show2(batch):
    plt.figure(figsize=(40,40)) 
    plt.imshow((make_grid(batch, 4)[:3]).permute(1,2,0))

class Normalization(nn.Module):
    def __init__(self, mean=torch.zeros(3), std=torch.ones(3)):
        super(Normalization, self).__init__()
        self.mean = nn.Parameter(mean.view(-1, 1, 1), requires_grad=False)
        self.std = nn.Parameter(std.view(-1, 1, 1), requires_grad=False)

    def forward(self, img):
        return (img - self.mean) / self.std


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

def gram_matrices(ims):
    return torch.stack([gram_matrix(im[None]) for im in ims])


def transferrer(model, target_layers):
    layers = model.module._modules['1'].features.children()
    new = nn.Sequential(model.module._modules['0'])
    i=0
    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        new.add_module(name, layer)
        if name == target_layers[-1]:
            return new


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class Aggregator:
    def __init__(self, model, style_layers, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.style_layers = style_layers
        self.model = transferrer(model, style_layers)
        self.total = []
        self.total2 = []
        self.n = 0

    @torch.no_grad()
    def vecs(self, ims):
        x = ims.to(device)
        r = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.style_layers:
                r.append(gram_matrices(x))
        return r
        
    @staticmethod
    def update(total, by, p=1):
        if len(total) == 0:
            total = [(x**p).sum(0) for x in by]
        else:
            for i in range(len(by)):
                total[i] += (by[i]**p).sum(0)
        return total

    def _step(self, by):
        self.total = self.update(self.total, by)
        self.total2 = self.update(self.total2, by, 2)
        self.n += by[0].shape[0]

    def step(self, ims):
        by = self.vecs(ims)
        self._step(by)

    def get_mss(self):
        mus = [x/self.n for x in self.total]
        stds = [torch.sqrt((snd/self.n) - fst**2) for fst, snd in zip(mus, self.total2)]
        return list(zip(mus, stds))
    
    
class Aff(nn.Module):
    """Moves one normal distribution to another."""
    def __init__(self, ms1, ms2):
        super().__init__()
        self.m1 = nn.Parameter(ms1[0])
        self.m2 = nn.Parameter(ms2[0])
        self.sq = nn.Parameter(ms1[1]/ms2[1])
        
    def forward(self, x):
        return (x-self.m1) * self.sq + self.m2
    
    
def vectorise(style):
    return [(m.reshape(m.shape[0]*m.shape[1]), s.reshape(s.shape[0]*s.shape[1])) for m, s in style]


def expand(model):
    W = model.features._modules['0'].weight.detach()
    b = model.features._modules['0'].bias.detach()
    model.features._modules['0'] = nn.Conv2d(5, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    model.features._modules['0'].weight[:,:3].data = W.data
    model.features._modules['0'].bias.data = b.data
    return model


def load_vgg(path, chan=3, out=5):
    model = models.vgg16(pretrained=False, progress=False)
    model.features._modules['0'] = nn.Conv2d(chan, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    model.classifier._modules['6'] = nn.Linear(in_features=4096, out_features=out, bias=True)
    model = nn.DataParallel(nn.Sequential(
        Normalization(torch.zeros(chan), torch.ones(chan)),
        model
    ))
    model.load_state_dict(torch.load(path))
    return model


def assess_transfer(transfer, classifier, data_path, dose_c, dose_s,
                    batch_size=4, image_size=512, samples=20):
    # either take some images or do aggregate
    data_c, data_s = map(lambda x: DataLoader(ImageDataset(data_path, image_size, train=False, doses=[x])),
                         [dose_c, dose_s])

    n = samples/batch_size
    for b_c, b_s in take(n, zip(iter(data_c), iter(data_s))):
        print(F.softmax(classifier(transfer(b_c, b_s)), 1))
    

    