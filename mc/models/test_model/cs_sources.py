from pygenn.genn_model import create_custom_current_source_class

import torch

from torchvision.datasets import MNIST

import numpy as np

test_source = create_custom_current_source_class(
    "test_source",
    param_names=["amplitude"],
    injection_code="""
        $(injectCurrent, ($(gennrand_uniform) * 2.0 - 1.0) * $(amplitude));
    """
)

mnist_dataset_train = MNIST(root="./mnist", train=True, download=True)
mnist_dataset_test = MNIST(root="./mnist", train=False, download=True)

train_input = np.reshape(mnist_dataset_train.data.numpy(), (60000, 784))
train_output = torch.nn.functional.one_hot(mnist_dataset_train.targets, num_classes=10).numpy()

mnist_step_source_model = create_custom_current_source_class(
    "mnist_source",
    param_names=["n_samples_set", "pop_size", "batch_size",
                 "T", "n_sample_lst"],
    var_name_types=[("idx", "int"), ("t_last", "scalar")],
    extra_global_params=[("data", "scalar*"),
                         ("sample_lst", "int*")],
    injection_code="""
        //increase idx every T
        
        if($(t) - $(t_last) >= $(T)){
            $(t_last) = $(t);
            $(idx)++;
        }
        
        // sample_lst contains indices for samples to be drawn
        // from the dataset, and these are drawn concurrently
        // in batches. It starts back from the beginning of the list
        // if the list length is exceeded. This means that
        // one batch can contain samples corresponding to
        // indices from both the end and the start of sample_lst if
        // the length of sample_lst is not divisible by the batch size.
        
        const int sample_id = $(sample_lst)[($(idx)*$(batch_size)+$(batch))%$(n_sample_lst)];
        
        // the original data is of size (n_sample, pop_size)
        // and is flattened c-style.
        const int data_id = $(pop_size) * sample_id + $(id);
        $(injectCurrent, $data[data_id]);
        
        // if you want to simulate multiple epochs, 
        // sample_lst should contain randomly shuffled versions
        // of [0,...,n_samples_set-1], concatenated together.
        // If you don't care about repeating the order of samples
        // in each epoch, you can also just provide one shuffled version of
        // [0,...,n_samples_set-1] (or even less if you are subsampling).
    """
)
