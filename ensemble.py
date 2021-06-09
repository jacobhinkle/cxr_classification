#import packages
import numpy as np
import torch
from torchvision.models import densenet
import train_densenet
from train_densenet import cxr_net
import torchvision.transforms as transforms
import mimic_cxr_jpg
from tqdm import tqdm
from train_densenet import cxr_net
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

device = 'cuda'

def get_predictions(X, Y, Ymask, model):
    Ymask = Ymask.to(device)
    X = X.type(torch.float32).to(device)
    Y = Y.type(torch.float32).to(device)
    logits = model(X)
    preds = torch.sigmoid(logits)
    return preds, X, Y, Ymask

def validate(model, val_loader, test_loader):
    model.eval()
    model = model.features.cuda()
    metrics = {}
    splits = [('val', val_loader), ('test', test_loader)]
    for i, (split, loader) in enumerate(splits):
        valbar = loader
        valbar = tqdm(valbar, desc=split, position=0, leave=False)
        Ypreds, Yactual = {}, {}
        for task in mimic_cxr_jpg.chexpert_labels:
            Ypreds[task], Yactual[task] = [], []
        for batch in valbar:
            with torch.no_grad():
                batchout = get_predictions(*batch, model)
                if batchout is None:
                    continue
                preds, X, Y, Ymask = batchout
            for i, task in enumerate(mimic_cxr_jpg.chexpert_labels):
                pred = preds[:, i].detach()
                mask = Ymask[:, i] == 1
                Yactual[task].append(Y[mask, i].cpu().numpy())
                Ypreds[task].append(pred[mask].cpu().numpy())
        # concatenate batch predictions
        for task in mimic_cxr_jpg.chexpert_labels:
            Ypreds[task] = np.concatenate(Ypreds[task], axis=0)
            Yactual[task] = np.concatenate(Yactual[task], axis=0)
        for task in mimic_cxr_jpg.chexpert_labels:
            Yp = Ypreds[task]
            Ya = Yactual[task]
            try:
                metrics[split + '_auc_' + task] = roc_auc_score(Ya, Yp)
            except ValueError:  # only one class predicted
                metrics[split + '_auc_' + task] = 0
    
    return metrics, Yp, Ya



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image-subdir', default='files',
            help='Subdirectory of datadir holding JPG files.')
    parser.add_argument('--num-folds', default=10, type=int,
             help='Number of folds in cross-validation')
    parser.add_argument('--fold', required=True, type=int,
             help='Which fold of cross-validation to use in training?')
    parser.add_argument('--random-state', default=0,  type=int,
             help='Random state to use in cross-validation')
    parser.add_argument('--batch-size', default=64, type=int,
            help='Batch size for test data loading')
    args = parser.parse_args()

    # Get train data
    train, val, test = mimic_cxr_jpg.cv(image_subdir=args.image_subdir, num_folds=args.num_folds, fold=args.fold,
                                  label_method={l:'zeros_uncertain_nomask' for l in mimic_cxr_jpg.chexpert_labels})
    print(train[0][1][0])

    val_loader = DataLoader(val,batch_size=args.batch_size,shuffle=False,num_workers=8,pin_memory=True,sampler=None)
    test_loader = DataLoader(test,batch_size=args.batch_size,shuffle=False,num_workers=8,pin_memory=True,sampler=None)
        
    # Load model
    model = cxr_net('densenet121', pretrained=True)
    model.load_state_dict(torch.load('/home/64f/cxr/cxr_classification/saved_models/256/model_epoch13.pt'))
    model.eval()
    model = model.features.cuda()
    for batch in val_loader:
        with torch.no_grad():
            batchout = get_predictions(*batch,model)
            if batchout is None:
                continue
            preds, X, Y, Ymask = batchout

    print(preds.shape)
    print(X[0].shape)
    print(Y.shape)
    print(Ymask.shape)

    '''
    metrics, Yp = validate(model,val_loader,test_loader)
    print(Yp[0])
    print(Ya.shape)
    print(Yp.shape)
    '''
 
