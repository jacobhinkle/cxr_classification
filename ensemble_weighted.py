import numpy as np
import torch
from torch import dstack
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

class SoftmaxWeightedEnsembleBinary(nn.Module):
    """
    Multi-label ensemble averaging of sigmoid logits.

    This model has a single parameter w which is a 2D tensor such that the
    weight for finding f and scale s is w[f, s]. This parametrization satisfies
    sum_s w[f, s] = 1.

    Given logits l[f, s], this provides a numerically stable way to compute
    logit(sum_s w[f, s] sigmoid(l[f, s])) for each finding f.

    Note that each label f is treated separately, even though everything is
    computed at once for the whole batch.

    When trained, torch.softmax(self.weight_l) gives the weights used to perform
    ensembling, and is interpreted as the importance of each scale, for
    predicting a given finding.

    Note that the weights are initialized to uniform.
    """
    def __init__(self, num_labels, num_learners):
        super().__init__()
        # logit parameterization of weights
        self.weight_l = nn.Parameter(torch.zeros(num_labels, num_learners))

    def forward(self, *xs):
        xs = torch.stack(xs, dim=2)
        # logsoftmax = x - LSE(x)
        logw = F.log_softmax(self.weight_l, dim=1)
        logp = F.logsigmoid(xs)
        # log probability of ensemble prediction
        # Note xs is 3D with dimensions [batch, label, scale], while logw is
        # [label, scale], so we unsqueeze a new dimension before adding
        logwp = logw[None, :, :] + logp  # equiv to log(w * p)
        # Ensemble probability is sumexp of logwp along scale dimension. Get log of that
        logp_ens = torch.logsumexp(logwp, dim=-1)
        # transform to logit instead of logprob
        # this is logit(p)=log(p/(1-p)) for p=exp(logp_ens)
        # PyTorch as of 1.9 doesn't have a softplusinv function, so we make one
        #logitp_ens = - torch.softplusinv(- logp_ens)
        logitp_ens = logp_ens - torch.log(- torch.expm1(logp_ens))
        return logitp_ens

if __name__ == '__main__':
    logits_256 = torch.load('logits/logits_256_val_fold0.pt', map_location='cpu')
    logits_512 = torch.load('logits/logits_512_val_fold0.pt', map_location='cpu')
    logits_1024 = torch.load('logits/logits_1024_val_fold0.pt', map_location='cpu')
    logits_2048 = torch.load('logits/logits_2048_val_fold0.pt', map_location='cpu')

    trainY = torch.load('logits/Y_256_val_fold0.pt', map_location='cpu')

    logits = [logits_256, logits_512, logits_1024, logits_2048]

    dataset = TensorDataset(*logits, trainY)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=8)

    num_labels = 14
    num_learners = len(logits)

    #model = LogisticRegressionModel(num_labels * num_learners, num_labels)
    model = SoftmaxWeightedEnsembleBinary(num_labels, num_learners)

    criterion = nn.BCEWithLogitsLoss()

    learning_rate = 1e1

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # decay learning rate exponentially
    # Half lifes of common choices for the rate, gamma, in epochs:
    #   0.9     7
    #   0.99    69
    #   0.999   693
    #   0.9999  6932
    lr_sched = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

    epbar = tqdm(range(100), desc='epoch')
    for epoch in epbar:
        # uncomment to monitor weights during training
        print("Current weights", torch.softmax(model.weight_l, dim=-1))
        eploss = 0.
        for i, batch in enumerate(dataloader):
            labels = batch[-1]
            optimizer.zero_grad()
            outputs = model(*batch[:-1])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            eploss += loss.item()
        lr_sched.step()
        epbar.set_postfix(loss=eploss / len(dataloader))

    # save the weights as a numpy array
    np.save('stacking_weights_fold0.npy', torch.softmax(model.weight_l, dim=-1).detach().cpu().numpy())
