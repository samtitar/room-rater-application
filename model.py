import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=6, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.lin1 = nn.Linear(128 * 4 * 4, 1024)
        self.lin2 = nn.Linear(1024, 256)
        self.lin3 = nn.Linear(256, 64)
        self.lin4 = nn.Linear(64, 11)

        # self.convolutional = nn.Sequential(
        #     # 256x256 to 65x65 to 32x32
        #     nn.Conv2d(3, 32, kernel_size=6, stride=4, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, 2),

        #     # 32x32 to 15x15 to 7x7
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, 2),
            
        #     # 7x7 to 4x4
        #     nn.Conv2d(64, 128, kernel_size=4, stride=1),
        #     nn.ReLU(inplace=True),
        # )

        # self.linear = nn.Sequential(
        #     # Fully connected
        #     nn.Linear(128 * 4 * 4, 1024),
        #     nn.ReLU(inplace=True),
            
        #     nn.Linear(1024, 256),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(256, 64),
        #     nn.ReLU(inplace=True),
            
        #     nn.Linear(256, 11)
        # )

    def forward(self, x):
        # x = self.convolutional(x)
        # x = x.view(x.size(0), -1)
        # return F.softmax(self.linear(x))

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))

        x = x.view(-1, 128 * 4 * 4)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)

        return x

