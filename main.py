from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler

import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''
data_dir = '/Users/sharon/data/EE148/'

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.1)
        self.dropout2 = nn.Dropout2d(0.1)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

        #Linear, Conv2d, MaxPool2d, AvgPool2d, ReLU, Softmax, BatchNorm2d, Dropout, Flatten, Sequential.

    def forward(self, x, get_feature_vector=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)

        if get_feature_vector:
            return x

        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    figure = plt.figure(figsize=(8, 8))
    plt.suptitle('Model Mistakes')
    cols, rows = 3, 3
    i = 1
    y_true = []
    y_pred = []
    data_np, feature_vector_np = None, None
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            feature_vector = model(data, get_feature_vector=True)
            if data_np is not None:
                data_np = np.append(data_np,
                        data.cpu().detach().numpy(), axis=0)
            else:
                data_np = data.cpu().detach().numpy()
            if feature_vector_np is not None:
                feature_vector_np = np.append(feature_vector_np,
                        feature_vector.cpu().detach().numpy(), axis=0)
            else:
                feature_vector_np = feature_vector.cpu().detach().numpy()


            for j, o in enumerate(output):
                output2 = torch.argmax(o)
                target2 = target[j]
                y_true.append(target2)
                y_pred.append(output2)
                if output2 != target2:
                    if i <= 9:
                        figure.add_subplot(rows, cols, i)
                        plt.title('T: %s, O: %s' % (target2.item(), output2.item()))
                        plt.axis("off")
                        plt.imshow(data[j].squeeze(), cmap="gray")
                        i += 1
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)
    plt.savefig('mismatches.png')
    plt.show()

    visualize_confusion_matrix(y_true, y_pred)
    visualize_embeddings(feature_vector_np, y_true)
    find_closest_vectors(data_np, feature_vector_np)
    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))

    return test_loss, correct / test_num


def find_closest_vectors(data_np, feature_vector_np):
    """
    Find the closest images by computing the Euclidean distance
    on the feature vectors.
    """
    figure = plt.figure(figsize=(8, 8))
    plt.suptitle("Images with Closest Feature Vectors")
    n = 4
    cols, rows = 9, n
    image_idx = 1
    for i in range(n):
        image = data_np[i,0,...]
        u = feature_vector_np[i]
        indices = [index for index, value in sorted(enumerate(feature_vector_np), key=lambda v:np.linalg.norm(u-v[1]))[:9]]
        for j in indices:
            figure.add_subplot(rows, cols, image_idx)
            if j == i:
                plt.title("Target")
            plt.axis("off")
            plt.imshow(data_np[j,0,...])
            image_idx += 1
    plt.savefig('closest.png')
    plt.show()


def visualize_embeddings(feature_vector_np, colors):
    """
    Visualize the test images in a 2-dimensional embedding space.
    """
    X_embedded = TSNE(n_components=2).fit_transform(feature_vector_np)
    classes = range(10)
    plt.title("Embeddings")
    scatter = plt.scatter(X_embedded[:,0], X_embedded[:,1], c=colors,
            cmap=plt.get_cmap('Spectral'))
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.savefig('embeddings.png')
    plt.show()


def visualize_confusion_matrix(y_true, y_pred):
    """
    Visualize the confusion matrix for the model.
    """
    C = confusion_matrix(y_true, y_pred)
    height, width = np.shape(C)
    fig, ax = plt.subplots()
    im = ax.imshow(C)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    # We want to show all ticks...
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    # ... and label them with the respective list entries
    ax.set_xticklabels(range(width))
    ax.set_yticklabels(range(height))

    # Loop over data dimensions and create text annotations.
    for i in range(height):
        for j in range(width):
            text = ax.text(j, i, C[i, j], ha="center", va="center", color="w")

    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    plt.savefig('confusion-matrix.png')
    plt.show()


def visualize_kernels(kernel, kernel2=None):
    """
    Visualize the first layer kernels and one kernel from the second layer.
    """
    figure = plt.figure(figsize=(8, 8))
    plt.suptitle('First Layer Kernels')
    cols, rows = 3, 3
    for i in range(1, 9):
        figure.add_subplot(rows, cols, i)
        plt.axis("off")
        plt.imshow(kernel[i-1].squeeze(), cmap="gray")

    figure.add_subplot(rows, cols, 9)
    plt.title("Kernel 1 of Conv2")
    plt.axis("off")
    plt.imshow(kernel2[0].squeeze(), cmap="gray")
    plt.savefig('kernels.png')
    plt.show()


def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--augmentation', type=bool, default=False,
                        help='augmentation for training (default: False)')
    parser.add_argument('--get-loglog-plot', type=bool, default=False,
                        help='plot loss vs. training size (default: False)')
    parser.add_argument('--model', type=str, default='fcNet',
                        help='augmentation for training (default: fcNet)')
    parser.add_argument('--training-set-divide', type=int, default=1, 
                        metavar='N',
                        help='divide training set by this number (default: 1)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        print(args.model)
        if args.model == 'Net':
            model = Net().to(device)
        elif args.model == 'ConvNet':
            model = ConvNet().to(device)
        else:
            model = fcNet().to(device)

        if args.get_loglog_plot:
            training_sizes = [60000 / i for i in [16, 8, 4, 2, 1]]
            model_names = ['mnist_model_div-%d.pt' % i for i in [16, 8, 4, 2, 1]]
            test_loss_list = []
            train_loss_list = [0.1710, 0.1422, 0.1134, 0.0554, 0.0511]
        else:
            model_names = [args.load_model]
        for model_name in model_names:
            model.load_state_dict(torch.load(model_name))

            kernel = model.conv1.weight.data.clone()
            kernel2 = model.conv2.weight.data.clone()
            visualize_kernels(kernel, kernel2)

            test_dataset = datasets.MNIST(data_dir, train=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]))

            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

            test_loss, test_acc = test(model, device, test_loader)
            if args.get_loglog_plot:
                test_loss_list.append(test_loss)

        if args.get_loglog_plot:
            if args.augmentation:
                loss_title = '%s (with Augmentation) Loss across Train Data Sizes' % args.model
                loss_figname = '%s-augmentation-loss-sizes' % args.model
            else:
                loss_title = '%s (without Augmentation) Loss across Train Data Sizes' % args.model
                loss_figname = '%s-no-augmentation-loss-sizes' % args.model

            plt.loglog(training_sizes, test_loss_list, label='test')
            plt.loglog(training_sizes, train_loss_list, label='train')
            plt.title(loss_title)
            plt.ylabel('Loss')
            plt.xlabel('Training Dataset Size')
            plt.legend()
            plt.tight_layout()
            plt.savefig('%s.png' % loss_figname)
            plt.show()

        return

    # Pytorch has default MNIST dataloader which loads data at each iteration
    if args.augmentation:
        print('augmentation')
        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                    transform=transforms.Compose([       # Data preprocessing
                        transforms.ToTensor(),           # Add data augmentation here
                        transforms.ColorJitter(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    else:
        print('no augmentation')
        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                    transform=transforms.Compose([       # Data preprocessing
                        transforms.ToTensor(),           # Add data augmentation here
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

    # Assign indices for disjoint training/validation or use a random subset for
    # training by using SubsetRandomSampler.
    training_set_size = len(train_dataset)
    if args.training_set_divide:
        training_set_size //= args.training_set_divide
    print('training_set_size: %d' % training_set_size)
    train_val_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=training_set_size)
    train_val_features, train_val_labels = next(iter(train_val_dataloader))
    train_val_indices = list(range(training_set_size))
    _, _, _, _, indices_train, indices_val = train_test_split(
            train_val_features, train_val_labels, train_val_indices,
            train_size=0.85, stratify=train_val_labels)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(indices_train)
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(indices_val)
    )

    # Load your model [fcNet, ConvNet, Net]
    print(args.model)
    if args.model == 'Net':
        model = Net().to(device)
    elif args.model == 'ConvNet':
        model = ConvNet().to(device)
    else:
        model = fcNet().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    test_loss_list = []
    test_acc_list = []
    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, device, val_loader)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        scheduler.step()    # learning rate scheduler

    if args.augmentation:
        loss_title = '%s (with Augmentation) Loss across Epochs' % args.model
        loss_figname = '%s-augmentation-loss' % args.model
        acc_title = '%s (with Augmentation) Accuracy across Epochs' % args.model
        acc_figname = '%s-augmentation-acc' % args.model
    else:
        loss_title = '%s (without Augmentation) Loss across Epochs' % args.model
        loss_figname = '%s-no-augmentation-loss' % args.model
        acc_title = '%s (without Augmentation) Accuracy across Epochs' % args.model
        acc_figname = '%s-no-augmentation-acc' % args.model

    if args.training_set_divide:
        loss_figname += '_div-%d' % args.training_set_divide
        acc_figname += '_div-%d' % args.training_set_divide

    plt.plot(test_loss_list)
    plt.title(loss_title)
    plt.xlabel('Epoch')
    plt.savefig('%s.png' % loss_figname)
    plt.show()

    plt.plot(test_acc_list)
    plt.title(acc_title)
    plt.xlabel('Epoch')
    plt.savefig('%s.png' % acc_figname)
    plt.show()

        # You may optionally save your model at each epoch here

    if args.save_model:
        model_name = "mnist_model"
        if args.training_set_divide:
            model_name += '_div-%d' % args.training_set_divide
        torch.save(model.state_dict(), "%s.pt" % model_name)


if __name__ == '__main__':
    main()
