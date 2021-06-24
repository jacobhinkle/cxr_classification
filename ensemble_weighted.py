import numpy as np
import torch
from torch import dstack
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

if __name__ == '__main__':
    logits_256 = torch.load('logits_256.pt').cpu()
    logits_512 = torch.load('logits_512.pt').cpu()
    logits_1024 = torch.load('logits_1024.pt').cpu()
    logits_2048 = torch.load('logits_2048.pt').cpu()

    testY_256 = torch.load('testY_256.pt').cpu()


    print(logits_256.shape, testY_256.shape)

    logits = [logits_256, logits_512, logits_1024, logits_2048]

    trainX = stacked_dataset(logits)
    trainY = testY_256
    print(trainX.shape)

    dataset = trainData(trainX, trainY)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=8)

    input_dim = 56
    output_dim = 14

    model = LogisticRegressionModel(input_dim, output_dim)

    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.001

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(100):
        running_loss = 0.0
        for i, (data, labels) in enumerate(dataloader):
            # Load images as Variable
            data = data.requires_grad_()
            labels = labels

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 500 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0








