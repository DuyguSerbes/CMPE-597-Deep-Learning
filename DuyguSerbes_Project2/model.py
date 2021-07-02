import torch.nn as nn
import torch.nn.functional as F

#CNN architecture
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # convolutional layer (sees 32x32x3 image tensor)
        self.conv1 = nn.Conv2d(3, 48, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(48)
        # convolutional layer (sees 16x16x48 tensor)
        self.conv2 = nn.Conv2d(48, 96, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(96)
        # dropout layer (p=0.1)
        self.dropout_cnn = nn.Dropout(0.1)
        # convolutional layer (sees 8x8x96 tensor)
        self.conv3 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(192)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (192 * 4 * 4 -> 512)
        self.fc1 = nn.Linear(192* 4 * 4, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        # dropout layer (p=0.3)
        self.dropout = nn.Dropout(0.3)
        # linear layer (512 -> 10)
        self.fc2 = nn.Linear(512, 10)



    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.dropout_cnn(x)
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        # flatten image input
        x = x.view(-1, 192 * 4 * 4)
        flatten=x
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1_bn(self.fc1(x)))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)

        return x,flatten

