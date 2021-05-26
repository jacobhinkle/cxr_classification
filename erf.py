import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from torchvision.models import densenet
from torch.utils.data import DataLoader
import mimic_cxr_jpg
import train_densenet
from train_densenet import cxr_net
from tqdm import tqdm

def erf(model, X):
    """
    Given a model that outputs a 2D image, do the following.

    If data has batch_size or num_channels greater than 1, we average over that
    dimension.
    """
    X.requires_grad_(True)
    Y = model(X)

    M, N = Y.shape[-2:]

    loss = Y[..., M//2, N//2].sum() # take middle pixel and sum over batch/chans

    loss.backward()

    # effective receptive field is the gradient variance image
    # we compute the uncentered sample variance
    sumsqgrad = (X.grad**2).sum(dim=0)

    return sumsqgrad.detach()

def erf_stream(model, dataloader, device='cuda'):
    es = None
    n = 0
    for X,Y,Z in tqdm(dataloader):
        X = X.to(device)
        e = erf(model, X).detach()
        if es is None:
            es = e
        else:
            es += e
        n += X.shape[0]
        del X
    es /= n
    return es

def fit_gaussian_moment(erf):
    """
    Fit a 2D Gaussian to the provided erf heatmap using second moments.

    Assumes the curve is centered on the middle pixel of the image and is
    isotropic. We return sigma and height.
    """
    X, Y = torch.meshgrid(
        torch.arange(erf.shape[-2]),
        torch.arange(erf.shape[-1]),
    )
    X = X.to(erf.device)
    Y = Y.to(erf.device)
    
    # only use last channel for computing moment
    erf = erf[[-1], ...]
    
    XY = torch.stack([X, Y], dim=0)
    
    esum = erf.sum()
    if esum == 0:
        print("erf sum is zero!")    
    
    mu = (erf * XY).sum(dim=[1,2]) / esum
        
    sqdist = (X - mu[0])**2 + (Y - mu[1])**2
    var = (erf * sqdist).mean() / erf.mean()
    sigma = torch.sqrt(var)

    # now use this sigma to estimate the L2-optimal height
    g = torch.exp((- 0.5 / var) * sqdist)
    # minimize sum|ht * g - erf|^2 => ht = sum(erf) / sum(g)
    height = erf.mean() / g.mean()

    return mu, sigma, height


def fit_gaussian_l2(erf):
    """
    Fit a 2D Gaussian to the provided erf heatmap using L2 distance.

    Assumes the curve is centered on the middle pixel of the image and is
    isotropic. We return sigma and height.

    This is initialized with the moment method.
    """
    sigma, height = fit_gaussian_moment(erf)

    raise NotImplementedError
    
    return sigma, height

def calculate_erf(model, X):
    if isinstance(X, torch.Tensor):
        e = erf(model, X)
    else: # assume dataloader
        e = erf_stream(model, X)
    
    mu, sig, ht = fit_gaussian_moment(e)
        
    return e,sig

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
    #Get test data
    train, val, test = mimic_cxr_jpg.cv(image_subdir=args.image_subdir, num_folds=args.num_folds, fold=args.fold, 
            random_state=args.random_state, 
            stratify=False,
            label_method={l:'zeros_uncertain_nomask' for l in mimic_cxr_jpg.chexpert_labels})


    data = DataLoader(test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            sampler=None,
        )
    
    #load model
    model = cxr_net('densenet121', pretrained=True)
    model.load_state_dict(torch.load('/home/64f/cxr/cxr_classification/out256x256/model_epoch18.pt'))
    model.eval()
    model = model.features.cuda()

    #imagenet pretrained model
    model_pretrained = cxr_net('densenet121', pretrained=True)    
    model_pretrained = model_pretrained.features.cuda()
    
    #untrained model
    model_untrained = cxr_net('densenet121', pretrained=False)
    model_untrained = model_untrained.features.cuda()
    
    e, sig = calculate_erf(model,data)
    #e, sig_pretrained = calculate_erf(model_pretrained,data)
    #e, sig_untrained = calculate_erf(model_untrained,data)
    print('ERF for trained model:', sig)
    #print('ERF for pretrained model:', sig_pretrained)
    #print('ERF for untrained model:', sig_untrained)

    #plot the ERF - only for trained model 
    mu, _, ht = fit_gaussian_moment(e)
    print(mu,ht)
    with torch.no_grad():
        emin = e.min()
        emax = e.max()
        #print(emax,e,emin)
        if e.shape[0] == 3:
            plt.imshow(((e - emin) / (emax - emin)).cpu().transpose(0, 2))
        else:
            plt.imshow(((e - emin) / (emax - emin)).cpu().squeeze(0)) 
        #c = mpl.patches.Circle(mu, sig, color='r', lw=1, fill=False)
        c = mpl.patches.Ellipse(mu, 2*ht, 2*sig, edgecolor='r', lw=1, facecolor="none")
        plt.gca().add_artist(c)
        plt.savefig('ERF-256.jpg',dpi=300)
   
