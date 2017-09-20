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
parser = argparse.ArgumentParser(description='PyTorch MNIST Association Learning')
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
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=600, metavar='N',
                    help='how many batches to wait before logging train/test status')
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


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(1152, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        print("0:", x.size())
        x = F.elu(self.conv1_1(x))
        x = F.elu(self.conv1_2(x))
        x = self.pool1(x)
        print("1:", x.size())

        x = F.elu(self.conv2_1(x))
        x = F.elu(self.conv2_2(x))
        x = self.pool2(x)
        print("2:", x.size())

        x = F.elu(self.conv3_1(x))
        x = F.elu(self.conv3_2(x))
        x = self.pool3(x)
        print("3:", x.size())

        x = x.view(x.size(0), -1)
        emb = F.elu(self.fc1(x))
        logits = self.fc2(emb)

        return logits, emb


model = Net()
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)


def train():
    for step in tqdm(range(num_steps)):
        unlabeled = unlabeled_loader.__iter__().__next__()[0]
        data, target = train_loader.__iter__().__next__()
        unlabeled, data, target = unlabeled.cuda(), data.cuda(), target.cuda()
        unlabeled, data, target = Variable(unlabeled), Variable(data), Variable(target)

        model.train()
        optimizer.zero_grad()
        logits, emb = model(data)
        softmax = F.log_softmax(logits)
        ce_loss = F.nll_loss(softmax, target)
        loss = ce_loss
        loss.backward()
        optimizer.step()

        if args.log_interval and step % args.log_interval == 0:
            pred = softmax.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()
            print('\nTrain:\tLoss: {:.4f}\tAccuracy: {}/{} ({:.2f}%)\tCE Loss: {:.6f}'.format(
                loss.data[0], correct, args.batch_size, 100. * correct / args.batch_size,
                ce_loss.data[0]))
            test()


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        _, softmax_cln, _, _ = model(data)
        test_loss += F.nll_loss(softmax_cln, target, size_average=False).data[0]  # sum up batch loss
        pred = softmax_cln.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print(' Test:\tLoss: {:.4f}\tAccuracy: {}/{} ({:.2f}%)\tLR: {:.4f}'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset),
        optimizer.param_groups[0]['lr']))

train()
print("\nOPTIMIZATION FINISHED!")
test()
