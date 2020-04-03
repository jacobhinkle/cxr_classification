import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_recall_curve, auc
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.models.resnet as tvresnet
from tqdm import tqdm

import mimic_cxr_jpg
from torch_nlp_models.meters import CSVMeter

from affine_augmentation import densenet

import os


class Trainer:
    def __init__(
        self,
        model,
        train_data,
        num_epochs,
        output_dir,
        batch_size=64,
        val_iters=None,
        val_data=None,
        test_data=None,
        lr=1e-3,
        device='cuda',
        progress = False,
    ):
        self.model = model
        self.num_epochs = num_epochs
        self.device = device
        self.val_iters = val_iters
        self.output_dir = output_dir

        self.progress = progress,

        self.train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        ) if val_data is not None else None
        self.test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        ) if test_data is not None else None

        self.epoch_meter = CSVMeter(os.path.join(self.output_dir, 'epoch_metrics.csv'), buffering=1)
        self.val_meter = CSVMeter(os.path.join(self.output_dir, 'val_metrics.csv'), buffering=1)
        self.iter_meter = CSVMeter(os.path.join(self.output_dir, 'iter_metrics.csv'))

        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.optim = optim.SGD(self.model.parameters(), lr=lr)

        self.total_iters = 0

    def train(self):
        self.epbar = range(self.num_epochs)
        self.epbar = tqdm(self.epbar, desc='epoch')
        for self._epoch in self.epbar:
            eploss = self.epoch()
            valmetrics = self.validate() if self.val_iters is None else {}
            self.epoch_meter.update(train_loss=eploss, **valmetrics)

    def epoch(self):
        self.itbar = self.train_loader
        self.itbar = tqdm(self.itbar, desc='iter')
        eploss = 0
        for self._iter, batch in enumerate(self.itbar):
            itloss = self.iteration(*batch)
            if itloss is None:
                continue
            self.iter_meter.update(loss=itloss)
            eploss += itloss / len(self.train_loader)
        return eploss

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
            valmetrics = self.validate()
            self.val_meter.update(**valmetrics)

        return loss.item()

    def validate(self):
        self.model.eval()
        metrics = {}
        splits = [('val', self.val_loader), ('test', self.test_loader)]
        for i, (split, loader) in enumerate(splits):
            valbar = loader
            if self.progress:
                valbar = tqdm(valbar, desc=split, position=len(splits)-i)
            valloss = 0
            Ypreds, Yactual = {}, {}
            for task in mimic_cxr_jpg.chexpert_labels:
                Ypreds[task], Yactual[task] = [], []
            for batch in valbar:
                with torch.no_grad():
                    batchout = self.batch_forward(*batch)
                    if batchout is None:
                        continue
                    preds, bce, loss, X, Y, Ymask = batchout
                for i, task in enumerate(mimic_cxr_jpg.chexpert_labels):
                    pred = preds[:, i].detach()
                    mask = Ymask[:, i] == 1
                    Yactual[task].append(Y[mask, i].cpu().numpy())
                    Ypreds[task].append(pred[mask].cpu().numpy())
                valloss += loss.detach().cpu().item()

            metrics[split + '_loss'] = valloss/len(valbar)

            for task in mimic_cxr_jpg.chexpert_labels:
                Yp = np.concatenate(Ypreds[task], axis=0)
                Ya = np.concatenate(Yactual[task], axis=0)

                ap = average_precision_score(Ya, Yp)
                metrics[split + '_avg_prec_' + task] = ap

                try:
                    metrics[split + '_auc_' + task] = roc_auc_score(Ya, Yp)
                except ValueError:  # only one class predicted
                    metrics[split + '_auc_' + task] = 0
        self.model.train()
        return metrics


if __name__ == '__main__':
    #model = tvresnet.resnet18(pretrained=True)
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datadir', '-d', default=mimic_cxr_jpg.topdir,
            help='Top-level directory of MIMIC-CXR-JPG dataset download.')
    parser.add_argument('--outputdir', '-o', required=True,
            help='Where to write outputs (trained weights, CSV of metrics)')
    parser.add_argument('--image-subdir', default='files',
            help='Subdirectory of datadir holding JPG files.')
    parser.add_argument('--epochs', default=100, type=int,
            help='Number of epochs to train for.')
    parser.add_argument('--val-iters', default=None, type=int,
            help='Compute validation metrics every this many iterations. None for once per epoch.')
    parser.add_argument('--batch-size', default=64, type=int,
            help='Batch size for SGD.')
    parser.add_argument('--learning-rate', default=1e-3, type=float,
            help='Learning rate for SGD.')
    parser.add_argument('--label-method', default='ignore_uncertain', choices=[
        'ignore_uncertain',
        'zeros_uncertain',
        'ones_uncertain',
        'three_class',
        'four_class',
        'missing_neg',
        ],
            help='Labelling method. "No Finding" will always use "missing_neg".')
    args = parser.parse_args()

    # Reproducibility
    # cf. https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

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
        datadir=args.datadir,
        image_subdir=args.image_subdir,
        label_method=args.label_method,
    )

    t = Trainer(model, train, args.epochs, args.outputdir,
        batch_size=args.batch_size,
        val_iters=args.val_iters,
        val_data=val,
        test_data=test,
        device=device,
    )

    t.train()
