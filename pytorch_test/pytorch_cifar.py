from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.002, metavar='LR',
                    help='learning rate (default: 0.002)')
parser.add_argument('--lr-decay-first', type=float, default=0.5, metavar='M',
                    help='LR decay start in (0,1) interval (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=None, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('/data', train=True, download=True, transform=transform),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('/data', train=False, transform=transform),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(num_features=96, affine=True)
        self.conv1_fn = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_features=96, affine=True)
        self.conv2_fn = nn.LeakyReLU(negative_slope=0.1)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(num_features=96, affine=True)
        self.conv3_fn = nn.LeakyReLU(negative_slope=0.1)
        self.pool1_bn = nn.BatchNorm2d(num_features=96, affine=True)

        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(num_features=192, affine=True)
        self.conv4_fn = nn.LeakyReLU(negative_slope=0.1)
        self.conv5 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(num_features=192, affine=True)
        self.conv5_fn = nn.LeakyReLU(negative_slope=0.1)
        self.conv6 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv6_bn = nn.BatchNorm2d(num_features=192, affine=True)
        self.conv6_fn = nn.LeakyReLU(negative_slope=0.1)
        self.pool2_bn = nn.BatchNorm2d(num_features=192, affine=True)

        self.conv7 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv7_bn = nn.BatchNorm2d(num_features=192, affine=True)
        self.conv7_fn = nn.LeakyReLU(negative_slope=0.1)
        self.conv8 = nn.Conv2d(192, 192, kernel_size=1, padding=0)
        self.conv8_bn = nn.BatchNorm2d(num_features=192, affine=True)
        self.conv8_fn = nn.LeakyReLU(negative_slope=0.1)
        self.conv9 = nn.Conv2d(192, 10, kernel_size=1, padding=0)
        self.conv9_bn = nn.BatchNorm2d(num_features=10, affine=True)
        self.conv9_fn = nn.LeakyReLU(negative_slope=0.1)
        self.pool3_bn = nn.BatchNorm2d(num_features=10, affine=False)

    def forward(self, x):
        x = self.conv1_fn(self.conv1_bn(self.conv1(x)))
        x = self.conv2_fn(self.conv2_bn(self.conv2(x)))
        x = self.conv3_fn(self.conv3_bn(self.conv3(x)))
        x = self.pool1_bn(F.max_pool2d(x, 2, stride=2))

        x = self.conv4_fn(self.conv4_bn(self.conv4(x)))
        x = self.conv5_fn(self.conv5_bn(self.conv5(x)))
        x = self.conv6_fn(self.conv6_bn(self.conv6(x)))
        x = self.pool2_bn(F.max_pool2d(x, 2, stride=2))

        x = self.conv7_fn(self.conv7_bn(self.conv7(x)))
        x = self.conv8_fn(self.conv8_bn(self.conv8(x)))
        x = self.conv9_fn(self.conv9_bn(self.conv9(x)))
        x = self.pool3_bn(F.avg_pool2d(x, kernel_size=x.size()[2:]))
        x = x.view(-1, 10)
        return F.log_softmax(x)


model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if args.log_interval and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def linear_lr_decay(epoch):
    decay_epochs = int(args.epochs * (1.0 - args.lr_decay_first))
    if epoch > args.epochs - decay_epochs:
        decay_epoch = epoch - (args.epochs - decay_epochs)
        factor = ((decay_epochs - (decay_epoch - 1)) / decay_epochs)
        lr = args.lr * factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


for epoch in tqdm(range(1, args.epochs + 1)):
    linear_lr_decay(epoch)
    train(epoch)
    test()
