import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from function import adaptive_instance_normalization as adain
from function import calc_mean_std, style_loss
from util import Normalization, expand


def vgg_enc(path=None, five=True):
    model = vgg16(pretrained=True, progress=False)
    if five: model = expand(model)
    d = 5 if five else 3 
    model.classifier = None  # to match regardless of num of classes
    model = nn.DataParallel(nn.Sequential(
        Normalization(torch.zeros(d), torch.ones(d)),
        model
    ))
    
    if path: model.load_state_dict(torch.load(path))
    return model.module

def to_encoder(classifier):
    fts = classifier[1].module.features
    return nn.Sequential(
        classifier[0],  # normalisation
        fts
    )

class BottleneckedAdaIN(nn.Module):
    def __init__(self, latent_dim, orignal_dim=512):
        self.down_m = nn.Linear(orignal_dim, latent_dim)
        self.down_s = nn.Linear(orignal_dim, latent_dim)
        self.up_m = nn.Linear(latent_dim, orignal_dim)
        self.up_s = nn.Linear(latent_dim, orignal_dim)

    def forward(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = calc_mean_std(style_feat)
        style_mean = self.up_m(self.down_m(style_mean))
        style_std = self.up_s(self.down_s(style_std))
        content_mean, content_std = calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)



class Transferrer(nn.Module):
    def __init__(self, encoder, decoder, bottleneck=None, alpha=1.0):
        super().__init__()
        enc_layers = list(encoder[1].features.children())
        self.encs = nn.ModuleList([
            nn.Sequential(encoder[0], *enc_layers[:2]),  # input -> relu1_1
            nn.Sequential(*enc_layers[2:7]),  # relu1_1 -> relu2_1
            nn.Sequential(*enc_layers[7:12]),  # relu2_1 -> relu3_1
            nn.Sequential(*enc_layers[12:19])  # relu3_1 -> relu4_1
        ])
        self.decoder = decoder
        self.alpha = alpha
        self.bottleneck = BottleneckedAdaIN(bottleneck) if bottleneck else adain

    def encode_(self, x):
        res = []
        for enc in self.encs:
            x = enc(x)
            res.append(x)
        return res

    def encode(self, x):
        for enc in self.encs:
            x = enc(x)
        return x

    def transfer(self, content, style):   
        style_feat = self.encode(style)
        content_feat = self.encode(content)
        t = self.bottleneck(content_feat, style_feat)
        t = self.alpha * t + (1 - self.alpha) * content_feat
        return self.decoder(t)

    def forward(self, content, style):
        style_feats = self.encode_(style)
        content_feat = self.encode(content)
        t = self.bottleneck(content_feat, style_feats[-1])
        t = self.alpha * t + (1 - self.alpha) * content_feat

        g_t = self.decoder(t)
        g_t_feats = self.encode_(g_t)

        # transfer might be needed for bottleneck, with pure adain it's just decode.encode
        half_consistency = F.l1_loss(transfer(content, content), content)

        loss_c = F.mse_loss(g_t_feats[-1], content_feat)  # TODO:  why not content feat? 
        loss_s = style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s, half_consistency


decoder = nn.Sequential(
    nn.Conv2d(512, 256, 3, padding=1),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='bilinear'),
    nn.Conv2d(256, 256, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(256, 256, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(256, 256, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(256, 128, 3, padding=1),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='bilinear'),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(128, 64, 3, padding=1),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='bilinear'),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 3, 3, padding=1),
)
