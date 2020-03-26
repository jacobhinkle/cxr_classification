import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.models.resnet as tvresnet
from tqdm import tqdm

import mimic_cxr_jpg

from affine_augmentation import densenet


class Trainer:
    def __init__(
        self,
        model,
        train_data,
        num_epochs,
        batch_size=64,
        val_iters=100,
        val_data=None,
        lr=1e-3,
        device='cuda',
    ):
        self.model = model
        self.num_epochs = num_epochs
        self.device = device
        self.val_iters = val_iters

        self.train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        if val_data is not None:
            self.val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.optim = optim.SGD(self.model.parameters(), lr=lr)

        self.total_iters = 0

    def train(self):
        self.epbar = range(self.num_epochs)
        self.epbar = tqdm(self.epbar, desc='epoch')
        for self._epoch in self.epbar:
            self.epoch()

    def epoch(self):
        self.itbar = self.train_loader
        self.itbar = tqdm(self.itbar, desc='iter')
        for self._iter, batch in enumerate(self.itbar):
            self.iteration(*batch)

    def batch_forward(self, X, Y, Ymask):
        weightsum = Ymask.sum().item()
        if weightsum == 0:
            return  # skip unlabelled batches

        X = X.type(torch.float32).to(device)
        Y = Y.type(torch.float32).to(device)
        Ymask = Ymask.to(device)

        preds = self.model(X)

        bce = self.criterion(preds, Y)
        loss = (bce * Ymask).sum() / weightsum
        return preds, bce, loss, X, Y, Ymask

    def iteration(self, *batch):
        self.optim.zero_grad()

        outputs = self.batch_forward(*batch)
        if outputs is None:
            return
        _, _, loss, _, _, _ = outputs

        self.itbar.set_postfix(loss=loss.item())

        loss.backward()

        self.optim.step()

        self.total_iters += 1

        if self.val_iters is not None and self.total_iters % self.val_iters == 0:
            self.validate()

        return loss.item()

    def validate(self):
        self.model.eval()
        for self._iter, batch in enumerate(self.itbar):
            preds, _, loss, _, Y, Ymask = self.batch_forward(*batch)
            # compute metrics based on preds, loss, Y, Ymask

        self.model.train()


if __name__ == '__main__':
    #model = tvresnet.resnet18(pretrained=True)
    print("Loading model")
    model = densenet.densenet121(
        pretrained=False,
        input_channels=1,
        num_classes=len(mimic_cxr_jpg.chexpert_labels),
    )
    print("Model loaded")

    nparams = sum([p.numel() for p in model.parameters()])
    print('num params', nparams)

    device='cuda'

    model = model.to(device)

    train, val, test = mimic_cxr_jpg.official_split(
        image_subdir='files256x256',
    )

    t = Trainer(model, train, 100, val_iters=1000, val_data=val, device=device)

    t.train()
