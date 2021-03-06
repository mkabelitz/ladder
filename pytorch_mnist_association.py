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
parser.add_argument('--labeled-samples', type=int, default=100, metavar='N',
                    help='number of labeled samples for training, None for all (default: 100)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=111, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=600, metavar='N',
                    help='how many batches to wait before logging train/test status')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])
    # [transforms.ToTensor()])

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
    print("index_set:", balanced_index_set)
    print("len train_loarder:", len(train_loader))
else:
    train_loader = torch.utils.data.DataLoader(mnist_tr_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               **kwargs)

unlabeled_loader = torch.utils.data.DataLoader(mnist_tr_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(mnist_te_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

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
        self.fc2 = nn.Linear(128, 10, bias=False)

    def forward(self, labeled, unlabeled):

        # pass for labeled batch
        x_l = F.elu(self.conv1_1(labeled))
        x_l = F.elu(self.conv1_2(x_l))
        x_l = self.pool1(x_l)
        x_l = F.elu(self.conv2_1(x_l))
        x_l = F.elu(self.conv2_2(x_l))
        x_l = self.pool2(x_l)
        x_l = F.elu(self.conv3_1(x_l))
        x_l = F.elu(self.conv3_2(x_l))
        x_l = self.pool3(x_l)
        x_l = x_l.view(x_l.size(0), -1)
        emb_l = F.elu(self.fc1(x_l))
        logits_l = self.fc2(emb_l)

        # pass for unlabeled batch
        x_u = F.elu(self.conv1_1(unlabeled))
        x_u = F.elu(self.conv1_2(x_u))
        x_u = self.pool1(x_u)
        x_u = F.elu(self.conv2_1(x_u))
        x_u = F.elu(self.conv2_2(x_u))
        x_u = self.pool2(x_u)
        x_u = F.elu(self.conv3_1(x_u))
        x_u = F.elu(self.conv3_2(x_u))
        x_u = self.pool3(x_u)
        x_u = x_u.view(x_u.size(0), -1)
        emb_u = F.elu(self.fc1(x_u))

        return logits_l, emb_l, emb_u


def get_visit_loss(p, weight=1.0):
    """Add the "visit" loss to the model.
    Args:
      p: [N, M] tensor. Each row must be a valid probability distribution
          (i.e. sum to 1.0)
      weight: Loss weight.
    """
    visit_probability = torch.sum(p, dim=0)
    t_nb = p.size()[1]
    tmp1 = Variable((torch.ones(t_nb) / t_nb).cuda())
    tmp2 = F.log_softmax(1e-8 + visit_probability)
    loss_fn = nn.KLDivLoss()
    visit_loss = loss_fn(input=tmp2, target=tmp1) * weight

    # print("visit_probability:\n", visit_probability)
    # print("t_nb:\n", t_nb)
    # print("tmp1:\n", tmp1)
    # print("tmp2:\n", tmp2)

    return visit_loss

    # Tensorflow original:
    #
    # visit_probability = tf.reduce_mean(
    #     p, [0], keep_dims=True, name='visit_prob')
    # t_nb = tf.shape(p)[1]
    # visit_loss = tf.losses.softmax_cross_entropy(
    #     tf.fill([1, t_nb], 1.0 / tf.cast(t_nb, tf.float32)),
    #     tf.log(1e-8 + visit_probability),
    #     weights=weight,
    #     scope='loss_visit')
    # return visit_loss


def get_semisup_loss(a, b, labels, walker_weight=1.0, visit_weight=1.0):
    """Add semi-supervised classification loss to the model.
    The loss constist of two terms: "walker" and "visit".
    Args:
      a: [N, emb_size] tensor with supervised embedding vectors.
      b: [M, emb_size] tensor with unsupervised embedding vectors.
      labels : [N] tensor with labels for supervised embeddings.
      walker_weight: Weight coefficient of the "walker" loss.
      visit_weight: Weight coefficient of the "visit" loss.
    """
    labels = labels.repeat(args.batch_size, 1)
    labels_transpose = torch.transpose(labels, 0, 1)
    equality_matrix = torch.eq(labels, labels_transpose).float()
    p_target = (equality_matrix / torch.sum(equality_matrix, dim=1).float())

    match_ab = torch.mm(a, torch.transpose(b, 0, 1))
    p_ab = F.softmax(1e-8 + match_ab)
    p_ba = F.softmax(1e-8 + torch.transpose(match_ab, 0, 1))
    p_aba = F.log_softmax(1e-8 + torch.mm(p_ab, p_ba))

    # we use KLDiv instead of cross entropy since cross entropy in Pytorch requires distinct class labels as targets
    loss_fn = nn.KLDivLoss()
    loss_aba = loss_fn(input=p_aba, target=p_target) * walker_weight
    visit_loss = get_visit_loss(p_ab, visit_weight)

    # print("emb labeled:\n", a)
    # print("emb unlabeled:\n", b)
    # print("labels:\n", labels)
    # print("labels transpose:\n", labels_transpose)
    # print("equality matrix:\n", equality_matrix)
    # print("p_target:\n", p_target)
    # print("match_ab:\n", match_ab)
    # print("p_ab:\n", p_ab)
    # print("p_ba:\n", p_ba)
    # print("p_aba:\n", p_aba)

    return loss_aba, visit_loss

    # Tensorflow original:
    #
    # match_ab = tf.matmul(a, b, transpose_b=True, name='match_ab')
    # p_ab = tf.nn.softmax(match_ab, name='p_ab')
    # p_ba = tf.nn.softmax(tf.transpose(match_ab), name='p_ba')
    # p_aba = tf.matmul(p_ab, p_ba, name='p_aba')
    #
    # loss_aba = tf.losses.softmax_cross_entropy(
    #     p_target,
    #     tf.log(1e-8 + p_aba),
    #     weights=walker_weight,
    #     scope='loss_aba')
    # visit_loss = add_visit_loss(p_ab, visit_weight)
    # return loss_aba, visit_loss

model = Net()
model.cuda()

# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
optimizer = optim.Adadelta(model.parameters(), weight_decay=0)


def train():
    for step in tqdm(range(num_steps)):
        unlabeled = unlabeled_loader.__iter__().__next__()[0]
        data, target = train_loader.__iter__().__next__()
        unlabeled, data, target = unlabeled.cuda(), data.cuda(), target.cuda()
        unlabeled, data, target = Variable(unlabeled), Variable(data), Variable(target)

        model.train()
        optimizer.zero_grad()
        logits, emb_l, emb_u = model(data, unlabeled)
        softmax = F.log_softmax(logits)
        ce_loss = F.nll_loss(softmax, target)
        loss_aba, visit_loss = get_semisup_loss(emb_l, emb_u, target)
        loss = ce_loss + (loss_aba + visit_loss)
        loss.backward()
        optimizer.step()

        if args.log_interval and step % args.log_interval == 0:
            pred = softmax.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()
            print('\nTrain:\tLoss: {:.4f}\tAccuracy: {}/{} ({:.2f}%)\tCE Loss: {:.6f}\tABA Loss: {:.6f}\tVisit Loss: {:.6f}'.format(
                loss.data[0], correct, args.batch_size, 100. * correct / args.batch_size,
                ce_loss.data[0], loss_aba.data[0], visit_loss.data[0]))
            print(model.conv1_1.bias.grad)
            test()


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        logits, emb, _ = model(data, data)
        softmax = F.log_softmax(logits)
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
