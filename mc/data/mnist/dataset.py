from torchvision.datasets import MNIST
from torch.nn.functional import one_hot

import os

BASE_FOLD = os.path.dirname(__file__)

dataset_train = MNIST(BASE_FOLD, download=True, train=True)
dataset_test = MNIST(BASE_FOLD, download=True, train=False)

input_train = dataset_train.data.flatten(start_dim=1).numpy() / 255.
output_train = one_hot(dataset_train.targets, num_classes=10).numpy()

input_test = dataset_test.data.flatten(start_dim=1).numpy() / 255.
output_test = one_hot(dataset_test.targets, num_classes=10).numpy()

