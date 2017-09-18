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
parser = argparse.ArgumentParser(description='PyTorch MNIST Gamma Network')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--labeled-samples', type=int, default=100, metavar='N',
                    help='number of labeled samples for training, None for all (default: 100)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.002, metavar='LR',
                    help='learning rate (default: 0.002)')
parser.add_argument('--lr-decay-first', type=float, default=0.67, metavar='M',
                    help='learning rate decay start in (0,1) interval (default: 0.67)')
parser.add_argument('--bn-momentum', type=float, default=0.9, metavar='M',
                    help='momentum for batch normalization (default: 0.9)')
parser.add_argument('--noise-std', type=float, default=0.3, metavar='M',
                    help='stddev for guassian noise (default: 0.3)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--train-log-interval', type=int, default=600, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test-log-interval', type=int, default=600, metavar='N',
                    help='how many batches to wait before logging test status')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

transform = transforms.Compose(
    # [transforms.ToTensor(),
    #  transforms.Normalize((0.1307,), (0.3081,))])
    [transforms.ToTensor()])

kwargs = {'num_workers': 0, 'pin_memory': True}

mnist_tr_dataset = datasets.MNIST('./torchvision_data', train=True, download=True, transform=transform)
mnist_te_dataset = datasets.MNIST('./torchvision_data', train=False, transform=transform)

if args.labeled_samples:
    balanced_index_set = []
    class_counts = [0] * 10
    overall_count = 0
    for i in range(mnist_tr_dataset.__len__()):
        if overall_count == args.labeled_samples:
            break
        cur_class = mnist_tr_dataset.__getitem__(i)[1]
        if class_counts[cur_class] < args.labeled_samples / 10:
            balanced_index_set.append(i)
            class_counts[cur_class] += 1
            overall_count += 1
    train_loader = torch.utils.data.DataLoader(mnist_tr_dataset,
                                               batch_size=args.batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(balanced_index_set),
                                               **kwargs)
else:
    train_loader = torch.utils.data.DataLoader(mnist_tr_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               **kwargs)

unlabeled_loader = torch.utils.data.DataLoader(mnist_tr_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(mnist_te_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

num_steps = args.epochs * len(unlabeled_loader)


class Noise(nn.Module):

    add_noise = False

    def __init__(self, shape, noise_std=args.noise_std):
        super().__init__()
        self.noise = Variable(torch.zeros(shape).cuda())
        self.std = noise_std

    def forward(self, x):
        if not Noise.add_noise:
            return x
        else:
            self.noise.data.normal_(0, std=self.std)
            # print(x.size(), self.noise.size())
            return x + self.noise


class RasmusBlock(nn.Module):
    def __init__(self, height_out, width_out, channels_out, act_fn, bn, noise, bias, scale):
        super().__init__()
        self.act_fn = act_fn
        self.bn_flag = bn
        self.noise_flag = noise
        self.bias_flag = bias
        self.scale_flag = scale
        if bn:
            self.bn = nn.BatchNorm2d(num_features=channels_out, affine=False, momentum=args.bn_momentum)
        if noise:
            self.noise = Noise((args.batch_size, channels_out, height_out, width_out))
        if bias:
            self.bias = nn.Parameter(torch.zeros((1, channels_out, 1, 1))).cuda()
        if scale:
            self.scale = nn.Parameter(torch.ones((1, channels_out, 1, 1))).cuda() if scale else None

    def forward(self, x):
        if self.bn_flag:
            x = self.bn(x)
        if self.noise_flag:
            x = self.noise(x)
        if self.bias_flag:
            x += self.bias
        if self.scale_flag:
            x = x * self.scale
        return self.act_fn(x)


class Conv2DBlock(RasmusBlock):
    def __init__(self, height_out, width_out, channels_in, channels_out, act_fn, kernel_size, padding,
                 bn=True, noise=True, bias=True, scale=False):
        super().__init__(height_out, width_out, channels_out, act_fn, bn, noise, bias, scale)
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return super().forward(self.conv(x))


class MaxPool2DBlock(RasmusBlock):
    def __init__(self, height_out, width_out, channels_out, act_fn=lambda x: x, kernel_size=2, stride=2,
                 bn=True, noise=True, bias=True, scale=True):
        super().__init__(height_out, width_out, channels_out, act_fn, bn, noise, bias, scale)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return super().forward(F.max_pool2d(x, self.kernel_size, self.stride))


class GlobalAvgPool2DBlock(RasmusBlock):
    def __init__(self, height_out, width_out, channels_out, act_fn=lambda x: x,
                 bn=True, noise=True, bias=True, scale=True):
        super().__init__(height_out, width_out, channels_out, act_fn, bn, noise, bias, scale)

    def forward(self, x):
        return super().forward(F.avg_pool2d(x, x.size()[2:]))


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.input_noise = Noise((args.batch_size, 1, 28, 28))
        self.conv1 = Conv2DBlock(28, 28, 1, 32, F.relu, 5, 2, bn=False)
        self.pool1 = MaxPool2DBlock(14, 14, 32)
        self.conv2 = Conv2DBlock(14, 14, 32, 64, F.relu, 3, 1)
        self.conv3 = Conv2DBlock(14, 14, 64, 64, F.relu, 3, 1)
        self.pool2 = MaxPool2DBlock(7, 7, 64)
        self.conv4 = Conv2DBlock(7, 7, 64, 128, F.relu, 3, 1)
        self.conv5 = Conv2DBlock(7, 7, 128, 10, F.relu, 1, 0)
        self.pool3 = GlobalAvgPool2DBlock(1, 1, 10)

        self.fc1 = nn.Linear(10, 10)
        self.fc1_bn = nn.BatchNorm1d(num_features=10, affine=False, momentum=args.bn_momentum)
        self.fc1_noise = Noise((args.batch_size, 10))
        self.fc1_bias = nn.Parameter(torch.zeros((1, 10))).cuda()
        self.fc1_scale = nn.Parameter(torch.ones((1, 10)).cuda())

        self.gamma_bn = nn.BatchNorm1d(num_features=10, affine=False, momentum=args.bn_momentum)

        self.a1 = nn.Parameter(torch.zeros((1, 10))).cuda()
        self.a2 = nn.Parameter(torch.ones((1, 10))).cuda()
        self.a3 = nn.Parameter(torch.zeros((1, 10))).cuda()
        self.a4 = nn.Parameter(torch.zeros((1, 10))).cuda()
        self.a5 = nn.Parameter(torch.zeros((1, 10))).cuda()
        self.a6 = nn.Parameter(torch.zeros((1, 10))).cuda()
        self.a7 = nn.Parameter(torch.ones((1, 10))).cuda()
        self.a8 = nn.Parameter(torch.zeros((1, 10))).cuda()
        self.a9 = nn.Parameter(torch.zeros((1, 10))).cuda()
        self.a10 = nn.Parameter(torch.zeros((1, 10))).cuda()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.input_noise(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)

        x = x.view(-1, 10)
        x = self.fc1_bn(self.fc1(x))
        z = self.fc1_noise(x)
        h = self.fc1_scale * (self.fc1_bias + z)

        if Noise.add_noise:
            u = self.gamma_bn(h)
            g_m = self.a1 * self.sigmoid(self.a2 * u + self.a3) + self.a4 * u + self.a5
            g_v = self.a6 * self.sigmoid(self.a7 * u + self.a8) + self.a9 * u + self.a10
            z_est = (z - g_m) * g_v + g_m
            z = z_est

        return F.log_softmax(h), z


model = Net()
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)


def linear_lr_decay(step):
    decay_steps = int(num_steps * (1.0 - args.lr_decay_first))
    if step > num_steps - decay_steps:
        decay_step = step - (num_steps - decay_steps)
        factor = ((decay_steps - (decay_step - 1)) / decay_steps)
        lr = args.lr * factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train():
    for step in tqdm(range(num_steps)):
        linear_lr_decay(step)
        unlabeled = unlabeled_loader.__iter__().__next__()[0]
        data, target = train_loader.__iter__().__next__()
        unlabeled, data, target = unlabeled.cuda(), data.cuda(), target.cuda()
        unlabeled, data, target = Variable(unlabeled), Variable(data), Variable(target)

        model.train()
        optimizer.zero_grad()
        Noise.add_noise = True
        softmax, _ = model(data)
        Noise.add_noise = True
        _, z_est = model(unlabeled)
        Noise.add_noise = False
        _, z = model(unlabeled)
        ce_loss = F.nll_loss(softmax, target)
        mse_loss = F.mse_loss(z, z_est)
        loss = ce_loss + mse_loss
        loss.backward()
        optimizer.step()

        if args.train_log_interval and step % args.train_log_interval == 0:
            pred = softmax.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()
            print('\nTrain:\tLoss: {:.4f}\tAccuracy: {}/{} ({:.2f}%)\tCE Loss: {:.6f}\tMSE Loss: {:.6f}'.format(
                ce_loss.data[0] + mse_loss.data[0], correct, args.batch_size, 100. * correct / args.batch_size,
                ce_loss.data[0], mse_loss.data[0]))
        if args.test_log_interval and step % args.test_log_interval == 0:
            # print(model.a1)
            # print(model.a2)
            # print(model.a3)
            # print(model.a4)
            # print(model.a5)
            # print(model.a6)
            # print(model.a7)
            # print(model.a8)
            # print(model.a9)
            # print(model.a10)

            # print(model.fc1_bias)
            # print(model.fc1_scale)

            print("GRAD:", model.fc1_scale.grad)

            test()


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        Noise.add_noise = False
        softmax, _ = model(data)
        test_loss += F.nll_loss(softmax, target, size_average=False).data[0]  # sum up batch loss
        pred = softmax.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print(' Test:\tLoss: {:.4f}\tAccuracy: {}/{} ({:.2f}%)\tLR: {:.4f}'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset),
        optimizer.param_groups[0]['lr']))

train()
print("\nOPTIMIZATION FINISHED!")
test()
