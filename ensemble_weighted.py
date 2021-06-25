import numpy as np
import torch
from torch import dstack
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

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
        logsoftmax = nn.LogSoftmax(dim=1)
        logsigmoid = nn.LogSigmoid()
        logw = logsoftmax(self.weight_l)
        logp = logsigmoid(xs)
        # log probability of ensemble prediction
        # Note xs is 3D with dimensions [batch, label, scale], while logw is
        # [label, scale], so we unsqueeze a new dimension before adding
        logwp = logw[None, :, :] + logp  # equiv to log(w * p)
        # transform to logit instead of logprob
        # this is logit(p)=log(p/(1-p)) for p=exp(logwp)
        logitwp = - torch.softplusinv(- logwp)
        return logitwp

if __name__ == '__main__':
    logits_256 = torch.load('logits_256.pt').cpu()
    logits_512 = torch.load('logits_512.pt').cpu()
    logits_1024 = torch.load('logits_1024.pt').cpu()
    logits_2048 = torch.load('logits_2048.pt').cpu()

    testY_256 = torch.load('testY_256.pt').cpu()


    print(logits_256.shape, testY_256.shape)

    logits = [logits_256, logits_512, logits_1024, logits_2048]

    trainY = testY_256

    #dataset = trainData(trainX, trainY)
    dataset = TensorDataset(*logits, trainY)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=8)

    num_labels = 14
    num_learners = len(logits)

    #model = LogisticRegressionModel(num_labels * num_learners, num_labels)
    model = SoftmaxWeightedEnsembleBinary(num_labels, num_learners)

    criterion = nn.BCEWithLogitsLoss()

    learning_rate = 0.001

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(100):
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            labels = batch[-1]
            optimizer.zero_grad()
            outputs = model(*batch[:-1])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 500 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

