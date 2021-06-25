import numpy as np
import torch
from torch import dstack
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

def stacked_dataset(members):
    stackX = None
    for logit in members:
        # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = logit
        else:
            stackX = dstack((stackX, logit))
    # flatten predictions to [rows, members x probabilities]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    return stackX

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

class trainData(Dataset):
    def __init__(self, logits, labels):
        self.samples = logits
        self.labels = labels

    def __len__(self):
        return int(self.samples.shape[0])

    def __getitem__(self, idx):
        X = self.samples[idx,:]
        y = self.labels[idx,:]
        return X,y

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
    logits_256 = torch.load('logits_256.pt', map_location='cpu')
    logits_512 = torch.load('logits_512.pt', map_location='cpu')
    logits_1024 = torch.load('logits_1024.pt', map_location='cpu')
    logits_2048 = torch.load('logits_2048.pt', map_location='cpu')

    trainY = torch.load('testY_256.pt', map_location='cpu')

    logits = [logits_256, logits_512, logits_1024, logits_2048]

    dataset = TensorDataset(*logits, trainY)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=8)

    num_labels = 14
    num_learners = len(logits)

    #model = LogisticRegressionModel(num_labels * num_learners, num_labels)
    model = SoftmaxWeightedEnsembleBinary(num_labels, num_learners)

    criterion = nn.BCEWithLogitsLoss()

    learning_rate = 1e1

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(100), desc='epoch'):
        print("Current weights", torch.softmax(model.weight_l, dim=-1))
        itbar = tqdm(enumerate(dataloader), desc='iter')
        for i, batch in itbar:
            labels = batch[-1]
            optimizer.zero_grad()
            outputs = model(*batch[:-1])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            itbar.set_postfix(loss=loss.item())
