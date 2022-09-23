import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os
import zipfile
from tqdm.notebook import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn, optim
from torchvision.models import densenet
import extract_features_dataloader 
from tqdm import tqdm 


def cxr_net(
        arch='densenet121',
        pretrained=False,
        num_classes=14,
    ):
    if 'densenet' in arch:

        if arch == 'densenet121':
            c = densenet.densenet121
            num_init_features = 64
        elif arch == 'densenet161':
            c = densenet.densenet161
            num_init_features = 96
        elif arch == 'densenet169':
            c = densenet.densenet169
            num_init_features = 64
        elif arch == 'densenet201':
            c = densenet.densenet201
            num_init_features = 64
        else:
            raise ValueError('arch must be one of: densenet121, densenet161, densenet169, densenet201')

        mod = c(pretrained=pretrained, num_classes=1000)
        # modify first conv to take proper input_channels
        oldconv = mod.features.conv0
        newconv = nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        newconv.weight.data = oldconv.weight.data.sum(dim=1, keepdims=True)
        mod.features._modules['conv0'] = newconv
        mod.classifier = nn.Linear(mod.classifier.in_features, num_classes)
        
    return mod

def get_features(n):
    def hook(model, input, output):
        feature[n] = output.detach()

    return hook

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datadir', '-d',
        default='/mnt/DGX01/Personal/4jh/cxr/MIMIC-CXR-JPG/',
        help='Location of data',
    )
    parser.add_argument(
        '--imagedir', '-i',
        default='files256x256',
        help='Location of input JPG images',
    )
    parser.add_argument(
        '--outdir', '-o',
        required=True,
        help='Location of output features', 
    )
    
    args = parser.parse_args()

    device = torch.device('cuda')
    model = cxr_net('densenet121')
    model.load_state_dict(torch.load('/mnt/DGX01/Personal/64f/cxr/MIMIC-CXR-JPG/model_epoch14.pt'))
    model = model.to(device)

    model.eval()

    common_args = dict(
            datadir=args.datadir,
            image_subdir=args.imagedir,
            return_studies = False,
            load_reports = False,
            load_activations = False

        )
    data = extract_features_dataloader.official_split(**common_args)
    dl = extract_features_dataloader.official_split(
        **common_args,
        dataloaders=True,
        dl_kwargs={
            'batch_size': 128,
            'num_workers': 12,
            },
    )

    print("...start extracting features...")
    with torch.no_grad():
        for i, (im,_,_,meta) in tqdm(enumerate(dl), total=len(dl)):
            im = im.type(torch.float32).to(device) 
            feature = {}
            model.features.register_forward_hook(get_features("feats"))
            out = model(im)
            fts = feature["feats"].squeeze()
            for j in range(fts.shape[0]):
                path = meta["path"][j].split('.')[0]
                torch.save(fts[j,:,:,:].detach().cpu(), args.outdir+str(path)+".pt")
            del feature
        









