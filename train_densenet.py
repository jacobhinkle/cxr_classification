import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_recall_curve, auc
import torch
from torch import nn, optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import mimic_cxr_jpg
from meters import CSVMeter
from networks import cxr_net

from datetime import datetime
import os

from contextlib import contextmanager
@contextmanager
def nvtxblock(desc):
    try:
        torch.cuda.nvtx.range_push(desc)
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def all_gather_vectors(tensors, *, device='cuda'):
    """
    All-gather 1D GPU tensors with heterogeneous lengths.
    """
    world_size = dist.get_world_size()

    assert isinstance(tensors, list)
    assert len(tensors) > 0

    # get the maximum length across all ranks
    hdls = []
    padded_tensors = []
    szst = []
    for t in tensors:
        szs = [torch.tensor(1).to(device) for _ in range(world_size)]
        dist.all_gather(szs, torch.tensor(t.shape[0]).to(device))
        szst.append(szs)
        maxlen = torch.tensor(szs).max().cpu().item()
        pts = [torch.zeros((maxlen,), device=device, dtype=t.dtype)
                for _ in range(world_size)]
        padded_tensors.append(pts)
        t_pad = torch.zeros_like(pts[0])
        t_pad[:t.shape[0]] = t
        hdls.append(dist.all_gather(pts, t_pad, async_op=True))
    # Now wait to complete and reassemble
    out = []
    for h, pts, szs in zip(hdls, padded_tensors, szst):
        h.wait()
        c = torch.cat([pt[:s] for s, pt in zip(szs, pts)], 0)
        out.append(c)
    return out

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
        distributed=False,
        amp=False,
        lr=0.0005,
        device='cuda',
        progress = False,
        reporter = True,
    ):
        self.model = model
        self.num_epochs = num_epochs
        self.device = device
        self.val_iters = val_iters
        self.output_dir = output_dir
        self.amp = amp
        self.lr = lr
        self.distributed = distributed
        self.progress = progress
        self.reporter = reporter

        if distributed:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

        self.train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=not distributed,
            num_workers=8,
            pin_memory=True,
            sampler=DistributedSampler(train_data, shuffle=True) if distributed else None,
        )
        self.val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            sampler=DistributedSampler(val_data) if distributed else None,
        ) if val_data is not None else None
        self.test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            sampler=DistributedSampler(test_data) if distributed else None,
        ) if test_data is not None else None

        if self.reporter:
            print(f"Number of minibatches in each split:"
                f" train {len(self.train_loader)}"
                f" val {len(self.val_loader)}"
                f" test {len(self.test_loader)}")

        if self.reporter:
            self.epoch_meter = CSVMeter(os.path.join(self.output_dir, 'epoch_metrics.csv'), buffering=1)
            self.val_meter = CSVMeter(os.path.join(self.output_dir, 'val_metrics.csv'), buffering=1)
            self.iter_meter = CSVMeter(os.path.join(self.output_dir, 'iter_metrics.csv'))

        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        #self.optim = optim.SGD(self.model.parameters(), lr=self.lr)

        self.total_iters = 0
        self.scaler = torch.cuda.amp.GradScaler()

    def train(self):
        self.epbar = range(self.num_epochs)
        validation_loss = []
        if self.progress and self.reporter:
            self.epbar = tqdm(self.epbar, desc='epoch', position=2)
        for self._epoch in self.epbar:
            with nvtxblock("Train Epoch"):
                eploss = self.epoch()
            if self.val_iters is None:
                with nvtxblock("Val Epoch"):
                    valmetrics = self.validate()
                if self.reporter:
                    for split, vm in valmetrics.items():
                        vm['split'] = split
                        self.val_meter.update(**vm)
            else:
                valmetrics = {}
            validation_loss.append(self.valLoss)
            if len(validation_loss) >= 3:
                if validation_loss[-1] >= validation_loss[-2] and validation_loss[-2] >= validation_loss[-3]:
                    self.lr = self.lr / 2
            elif len(validation_loss) >= 10:
                if validation_loss[-1] >= validation_loss[-10]:
                    break
            if self.reporter:
                self.update_meter(self.epoch_meter, valmetrics,
                        train_loss=eploss)
                # flush all meters at least once per epoch
                self.epoch_meter.flush()
                self.val_meter.flush()
                self.iter_meter.flush()

    def epoch(self):
        if self.reporter and not self.progress:
            print(f"Starting epoch {self._epoch} of {self.num_epochs}")
            epoch_start = datetime.now()
        self.itbar = self.train_loader
        if self.progress and self.reporter:
            self.itbar = tqdm(self.itbar, desc='iter', position=1, leave=False)
        eploss = 0
        for self._iter, batch in enumerate(self.itbar):
            with nvtxblock("Train Iteration"):
                itloss = self.iteration(*batch)
            if itloss is None:
                continue
            if self.reporter:
                self.iter_meter.update(loss=itloss)
            eploss += itloss / len(self.train_loader)
        if self.reporter and not self.progress:
            epoch_time = datetime.now() - epoch_start
            print(f"Epoch time: {epoch_time}")
        if self.reporter:
            torch.save(self.model.state_dict(), self.output_dir + f'/model_epoch{self._epoch}.pt')
        return eploss

    def batch_forward(self, X, Y, Ymask):
        Ymask = Ymask.to(device)
        weightsum = Ymask.sum()
        if self.distributed:
            # start reducing the number of weights in background early
            with torch.no_grad():
                weightsync_hdl = dist.all_reduce(weightsum, async_op=True)

        X = X.type(torch.float32).to(device).contiguous()
        Y = Y.type(torch.float32).to(device)

        preds = self.model(X)

        if self.distributed:
            weightsync_hdl.wait()
        weightsum = weightsum.item()
        if weightsum == 0:
            # if _reduced_ weightsum is zero, then entire minibatch (every
            # microbatch) holds no actual labels, and we can skip this
            # iteration. Otherwise, all ranks must continue to participate in
            # order to avoid a deadlock on the backward pass.
            return

        bce = self.criterion(preds, Y)
        loss = (bce * Ymask).sum() / weightsum
        return preds, bce, loss, X, Y, Ymask

    def iteration(self, *batch):
        self.optim.zero_grad()

        with nvtxblock("Forward"):
            if self.amp:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs = self.batch_forward(*batch)
            else:
                outputs = self.batch_forward(*batch)
        if outputs is None:
            return
        _, _, loss, _, _, _ = outputs

        if self.progress and self.reporter:
            self.itbar.set_postfix(loss=loss.item())

        with nvtxblock("Backward"):
            if self.amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

        with nvtxblock("Optim Step"):
            if self.amp:
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                self.optim.step()

        self.total_iters += 1

        if self.val_iters is not None and self.total_iters % self.val_iters == 0:
            val_start = datetime.now()
            with nvtxblock("Val"):
                valmetrics = self.validate()
            elapsed_time = datetime.now() - val_start
            if self.reporter:
                print(f"Validation time: {elapsed_time}")
                print(f"Learning rate: {self.lr}")
                self.update_meter(self.val_meter, valmetrics)

        return loss.item()

    def update_meter(self, meter, metrics, **addl_entries):
        for split, splitdict in metrics.items():
            splitloss = splitdict['loss']
            for metric, findingmets in splitdict['metrics'].items():
                meter.update(
                    **findingmets,
                    metric=metric,
                    split=split,
                    loss=splitloss,  # repeats the loss for each metric...
                    **addl_entries,
                )

    def validate(self):
        """Return a nested dict describing metrics

        {
          val:
            loss: 0.0309523509
            metrics:
              AUC:
                Atelectasis: 0.758395039
                Cardiomegaly: 0.8593293043
                  ...
              AveragePrecision:
                Atelectasis: 0.534251225
                Cardiomegaly: 0.753839493
                  ...
          test:
            ...
        }
        """
        if self.reporter:
            print("Computing validation and test metrics")
        self.model.eval()
        metrics = {}
        splits = [('val', self.val_loader), ('test', self.test_loader)]
        for i, (split, loader) in enumerate(splits):
            valbar = loader
            if self.progress and self.reporter:
                valbar = tqdm(valbar, desc=split, position=0, leave=False)
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
            # concatenate batch predictions
            for task in mimic_cxr_jpg.chexpert_labels:
                Ypreds[task] = np.concatenate(Ypreds[task], axis=0)
                Yactual[task] = np.concatenate(Yactual[task], axis=0)

            if self.distributed:
                allvectors = [torch.tensor(Ypreds[t]).to(device).contiguous()
                            for t in mimic_cxr_jpg.chexpert_labels] \
                        + [torch.tensor(Yactual[t]).to(device).contiguous()
                            for t in mimic_cxr_jpg.chexpert_labels]
                gathered = all_gather_vectors(allvectors, device=self.device)
                for i, task in enumerate(mimic_cxr_jpg.chexpert_labels):
                    Ypreds[task] = gathered[i]
                    Yactual[task] = gathered[i +
                            len(mimic_cxr_jpg.chexpert_labels)]

            metrics[split] = {'loss': valloss/len(valbar)}
            #metrics[split + '_loss'] = valloss/len(valbar)
            if split == 'val':
                self.valLoss =  valloss/len(valbar)

            metrics[split]['metrics'] = {'AUC': {}, 'AveragePrecision': {}}

            for task in mimic_cxr_jpg.chexpert_labels:
                Yp = Ypreds[task]
                Ya = Yactual[task]

                ap = average_precision_score(Ya, Yp)
                #metrics[split + '_avg_prec_' + task] = ap
                metrics[split]['metrics']['AveragePrecision'][task] = ap

                try:
                    auc = roc_auc_score(Ya, Yp)
                except ValueError:  # only one class predicted
                    auc = 0
                metrics[split]['metrics']['AUC'][task] = auc
        self.model.train()
        return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datadir', '-d', default=mimic_cxr_jpg.topdir,
            help='Top-level directory of MIMIC-CXR-JPG dataset download.')
    parser.add_argument(
        "--dicom_id_file",
        help="Restrict to only the dicom_ids in the 'dicom_id' column of a given CSV file.",
    )
    parser.add_argument('--outputdir', '-o', required=True,
            help='Where to write outputs (trained weights, CSV of metrics)')
    parser.add_argument('--image-subdir', default='files',
            help='Subdirectory of datadir holding JPG files.')
    parser.add_argument('--arch', default='densenet121', choices=['densenet121',
        'densenet161', 'densenet169', 'densenet201', 'msd100','resnet50'],
            help='Densenet architecture.')
    parser.add_argument('--feature_net_checkpoint', help='If given, load pretrained feature network from this checkpoint file.')
    parser.add_argument('--from-scratch', action='store_true',
            help='Do not initialize with ImageNet pretrained weights.')
    parser.add_argument('--epochs', default=100, type=int,
            help='Number of epochs to train for.')
    parser.add_argument('--val-iters', default=None, type=int,
            help='Compute validation metrics every this many iterations. None for once per epoch.')
    parser.add_argument('--batch-size', default=64, type=int,
            help='Batch size for SGD.')
    parser.add_argument('--learning-rate', default=1e-3, type=float,
            help='Learning rate for SGD.')
    parser.add_argument('--amp', action='store_true',
            help='Use automatic mixed precision (AMP).')
    parser.add_argument('--num-folds', default=10, type=int,
            help='Number of folds in cross-validation')
    parser.add_argument('--fold', required=True, type=int,
            help='Which fold of cross-validation to use in training?')
    parser.add_argument('--random-state', default=0,  type=int,
            help='Random state to use in cross-validation')
    parser.add_argument('--hide-progress', action='store_true',
            help='Do not display progress bar.')
    parser.add_argument('--single-node-data-parallel', action='store_true',
            help='Use torch.nn.DataParallel')
    parser.add_argument('--distributed-data-parallel', action='store_true',
            help='Use torch.distributed for multi-node parallelism')
    parser.add_argument('--label-method', default='ignore_uncertain', choices=[
        'ignore_uncertain',
        'zeros_uncertain',
        'ones_uncertain',
        'zeros_uncertain_nomask',
        'ones_uncertain_nomask',
        'three_class',
        'four_class',
        'missing_neg',
        ],
            help='Labeling method. "No Finding" will always use "missing_neg".')
    args = parser.parse_args()

    # Reproducibility
    # cf. https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    if args.arch[:3] == 'msd':
        from msd_classifier import MSDClassifier2d
        if args.arch == 'msd100':
            model = MSDClassifier2d(1, len(mimic_cxr_jpg.chexpert_labels), depth=100, maxdil=10, width=1)
    else:
        print(f"Initializing {args.arch} model with pretrained={not args.from_scratch}")
        model = cxr_net(args.arch, pretrained=not args.from_scratch)

    if args.feature_net_checkpoint is not None:
        print(f"Loading feature network state_dict from {args.feature_net_checkpoint}")
        sd = torch.load(args.feature_net_checkpoint, map_location='cpu')
        model.features.load_state_dict(sd)

    nparams = sum([p.numel() for p in model.parameters()])

    if args.single_node_data_parallel and args.distributed_data_parallel:
        raise Exception("Max one of distributed or single-node data parallel can be requested.")

    train, val, test = mimic_cxr_jpg.cv(
        args.num_folds,
        args.fold,
        datadir=args.datadir,
        dicom_id_file=args.dicom_id_file,
        image_subdir=args.image_subdir,
        random_state=args.random_state,
        stratify=False,
        #label_method={l:'zeros_uncertain_nomask' for l in mimic_cxr_jpg.chexpert_labels},
    )
    sampler = None

    if args.distributed_data_parallel:
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    else:
        world_size = 1
        local_rank = 0
        rank = 0

    # We do not use local_rank since we are now using -r6 -a1 -g1 -c7 on summit
    gpunum = local_rank

    device = torch.device('cuda', gpunum)
    model = model.to(device)

    if args.single_node_data_parallel:
        model = nn.DataParallel(model)
    elif args.distributed_data_parallel:
        dist.init_process_group('nccl')

        args.learning_rate *= world_size

        model = DDP(
            model,
            device_ids=[gpunum],
            output_device=gpunum,
        )

    t = Trainer(model, train, args.epochs, args.outputdir,
        batch_size=args.batch_size,
        val_iters=args.val_iters,
        val_data=val,
        test_data=test,
        progress=not args.hide_progress,
        reporter=rank == 0,
        device=device,
        amp=args.amp,
        distributed=args.distributed_data_parallel,
    )

    t.train()
