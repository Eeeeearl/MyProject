import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from tensorboardX import SummaryWriter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# define model parameters
NUM_EPOCHS = 90
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227
NUM_CLASSES = 1000
DEVICE_IDS = [0]
# modify this to point to your data directory
INPUT_ROOT_DIR = 'alexnet_data_in'
TRAIN_IMG_DIR = 'alexnet_data_in/imagenet'
OUTPUT_DIR = 'alexnet_data_out'
LOG_DIR = OUTPUT_DIR + '/tblogs'
CHECKPOINT_DIR = OUTPUT_DIR + '/models'

# make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class AlexNet(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """
    def __init__(self, num_classes=1000):
        """
        Define and allocate layers for this neural net.
        Args:
             num_classes(int): number of classes to predict with this model
        """
        super().__init__()
        # input size: (b x 3 x 227 x 227)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # output size: b x 96 x 55 x 55
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # LRN（局部归一化处理），后出现了BN（块归一化处理）
            nn.MaxPool2d(kernel_size=3, stride=2),  # output size: b x 96 x 27 x 27
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),  # output: bx256 x 27 x 27
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output size: b x 256 x 13 x 13
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # output: bx384 x 13 x 13
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),  # output: bx384 x 13 x 13
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # output: bx256 x 13 x 13
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output size: b x 256 x 6 x 6
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant(layer.bias, 0)
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 256 * 6 * 6)
        return self.fc(x)


if __name__ == '__main__':
    seed = torch.initial_seed()
    print('Used seed : {}'.format(seed))

    alexnet = AlexNet(num_classes=NUM_CLASSES).to(device)

    alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=DEVICE_IDS)
    print(alexnet)
    print('AlexNet created')

    # create dataset and data loader
    dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transforms.Compose([
        transforms.CenterCrop(IMAGE_DIM),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    print('Dataset created')

    dataloader = data.DataLoader(
        dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE
    )
    print('Dataloader created')

    optimizer = optim.Adam(params=alexnet.parameters(), lr=0.0001)
    print('Optimizer created')

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print('LR Scheduler created')

    print('Starting training...')
    total_steps = 1
    for epoch in range(NUM_EPOCHS):
        lr_scheduler.step()
        for imgs, classes in dataloader:
            imgs, classes = imgs.to(device), classes.to(device)

            # calculate the loss
            output = alexnet(imgs)
            loss = F.cross_entropy(output, classes)

            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if total_steps % 10 == 0:
                with torch.no_grad():
                    _, preds = torch.max(output, 1)
                    accuracy = torch.sum(preds == classes)

                    print('Epoch: {} \t Step: {} \t Loss: {:.4f} \t Acc: {}'
                          .format(epoch+1, total_steps, loss.item(), accuracy.item()))
                    # add tensorboard to draw the pic

            if total_steps % 100 == 0:
                with torch.no_grad():
                    print('*' * 10)
                    for name, parameter in alexnet.named_parameters():
                        if parameter.grad is not None:
                            avg_grad = torch.mean(parameter.grad)
                            print('\t{} - grad_avg: {}'.format(name, avg_grad))
                            # add tensorboard to draw the pic
                        if parameter.data is not None:
                            avg_weight = torch.mean(parameter.data)
                            print('\t{} - param_avg: {}'.format(name, avg_weight))
                            # add tensorboard to draw the pic

            total_steps += 1

        # save checkpoints
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epoch + 1))
        state = {
            'epoch': epoch,
            'total_steps': total_steps,
            'optimizer': optimizer.state_dict(),
            'model': alexnet.state_dict(),
            'seed': seed,
        }
        torch.save(state, checkpoint_path)


