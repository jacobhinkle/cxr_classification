#import packages
import numpy as np
import pandas as pd
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
import csv

device = 'cuda'

def get_data(X, Y, Ymask, model):
    Ymask = Ymask.to(device)
    X = X.type(torch.float32).to(device).contiguous()
    Y = Y.type(torch.float32).to(device)
    logits = model(X)
    preds = torch.sigmoid(logits)
    return preds, X, Y, Ymask

def pred(test_loader, model):
    model.eval()
    valbar = test_loader
    valbar = tqdm(valbar, position=0, leave=False)
    prediction, label, mask = [], [], []
    for batch in valbar:
        with torch.no_grad():
            batchout = get_data(*batch, model)
            if batchout is None:
                continue
            preds, X, Y, Ymask = batchout
            prediction.append(preds)
            label.append(Y)
            mask.append(Ymask)
    prediction = torch.cat(prediction,0)
    label = torch.cat(label,0)
    mask = torch.cat(mask,0)
    return prediction, label, mask

def auc_score(preds, Y, Ymask):
    Ypreds, Yactual, metrics = {}, {}, {}
    for task in mimic_cxr_jpg.chexpert_labels:
        Ypreds[task], Yactual[task] = [], []
    for i, task in tqdm(enumerate(mimic_cxr_jpg.chexpert_labels)):
        pred = preds[:, i].detach()
        mask = Ymask[:, i] == 1
        Yactual[task].append(Y[mask, i].cpu().numpy())
        Ypreds[task].append(pred[mask].cpu().numpy())
    
    for task in mimic_cxr_jpg.chexpert_labels:
        Ypreds[task] = np.concatenate(Ypreds[task], axis=0)
        Yactual[task] = np.concatenate(Yactual[task], axis=0)

    for task in mimic_cxr_jpg.chexpert_labels:
        Yp = Ypreds[task]
        Ya = Yactual[task]
        
        metrics[task + '_AUC'] = roc_auc_score(Ya, Yp)
    
    return metrics

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-folds', default=10, type=int,
             help='Number of folds in cross-validation')
    parser.add_argument('--fold', required=True, type=int,
             help='Which fold of cross-validation to use in training?')
    parser.add_argument('--random-state', default=0,  type=int,
             help='Random state to use in cross-validation')
    args = parser.parse_args()

    train_256, val_256, test_256 = mimic_cxr_jpg.cv(image_subdir="/mnt/DGX01/Personal/4jh/cxr/MIMIC-CXR-JPG/files256x256/", 
                                    num_folds=args.num_folds, fold=args.fold,
                                    label_method={l:'zeros_uncertain_nomask' for l in mimic_cxr_jpg.chexpert_labels})

    train_512, val_512, test_512 = mimic_cxr_jpg.cv(image_subdir="/mnt/DGX01/Personal/4jh/cxr/MIMIC-CXR-JPG/files512x512/", 
                                    num_folds=args.num_folds, fold=args.fold,
                                    label_method={l:'zeros_uncertain_nomask' for l in mimic_cxr_jpg.chexpert_labels})

    train_1024, val_1024, test_1024 = mimic_cxr_jpg.cv(image_subdir="/mnt/DGX01/Personal/4jh/cxr/MIMIC-CXR-JPG/files1024x1024/", 
                                    num_folds=args.num_folds, fold=args.fold,
                                    label_method={l:'zeros_uncertain_nomask' for l in mimic_cxr_jpg.chexpert_labels})

    train_2048, val_2048, test_2048 = mimic_cxr_jpg.cv(image_subdir="/mnt/DGX01/Personal/4jh/cxr/MIMIC-CXR-JPG/files2048x2048/", 
                                    num_folds=args.num_folds, fold=args.fold,
                                    label_method={l:'zeros_uncertain_nomask' for l in mimic_cxr_jpg.chexpert_labels})

    val_loader_256 = DataLoader(val_256,batch_size=512,shuffle=False,num_workers=8,pin_memory=True,sampler=None)
    test_loader_256 = DataLoader(test_256,batch_size=512,shuffle=False,num_workers=8,pin_memory=True,sampler=None)

    val_loader_512 = DataLoader(val_512,batch_size=256,shuffle=False,num_workers=8,pin_memory=True,sampler=None)
    test_loader_512 = DataLoader(test_512,batch_size=256,shuffle=False,num_workers=8,pin_memory=True,sampler=None)

    val_loader_1024 = DataLoader(val_1024,batch_size=110,shuffle=False,num_workers=8,pin_memory=True,sampler=None)
    test_loader_1024 = DataLoader(test_1024,batch_size=110,shuffle=False,num_workers=8,pin_memory=True,sampler=None)

    val_loader_2048 = DataLoader(val_2048,batch_size=30,shuffle=False,num_workers=8,pin_memory=True,sampler=None)
    test_loader_2048 = DataLoader(test_2048,batch_size=30,shuffle=False,num_workers=8,pin_memory=True,sampler=None)


    model_path = ['/home/64f/cxr/cxr_classification/saved_models/256/model_epoch14.pt',
                  '/home/64f/cxr/cxr_classification/saved_models/512/model_epoch18.pt',
                  '/home/64f/cxr/cxr_classification/saved_models/1024/model_epoch19.pt',
                  '/home/64f/cxr/cxr_classification/saved_models/2048/model_epoch21.pt']
        
    # Load model and get the  AUC score
    
    model_256 = cxr_net('densenet121', pretrained=True)
    model_256.load_state_dict(torch.load(model_path[0]))
    model_256.to("cuda")
    preds_256, Y_256, Ymask_256 = pred(test_loader_256, model_256)
    
    model_512 = cxr_net('densenet121', pretrained=True)
    model_512.load_state_dict(torch.load(model_path[1]))
    model_512.to("cuda")
    preds_512, Y_512, Ymask_512 = pred(test_loader_512, model_512)
    
    model_1024 = cxr_net('densenet121', pretrained=True)
    model_1024.load_state_dict(torch.load(model_path[2]))
    model_1024.to("cuda")
    preds_1024, Y_1024, Ymask_1024 = pred(test_loader_1024, model_1024)
    
    model_2048 = cxr_net('densenet121', pretrained=True)
    model_2048.load_state_dict(torch.load(model_path[3]))
    model_2048.to("cuda")
    preds_2048, Y_2048, Ymask_2048 = pred(test_loader_2048, model_2048)
    

    preds = (preds_256 + preds_512 + preds_1024 + preds_2048) / 4
    print(preds[0])
    
    metrics = auc_score(preds, Y_256, Ymask_2048)
    print(metrics)
    
    df = pd.DataFrame.from_dict(metrics, orient="index")
    df.to_csv("metrics.csv")

