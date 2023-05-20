import matplotlib.pyplot as plt

import numpy as np


import torch
from torchvision.datasets import MNIST

from mc.network import Network

from models import change_detection_spikes_yinyang as network_model
from models.cs_sources import step_source_model as stp_src

from ...misc.utils import plot_spike_times, calc_loss_interp, calc_class_acc_interp

import os

TASK_BASE_FOLD = os.path.dirname(__file__)

OUT_MIN = 0.1
OUT_MAX = 0.9

mnist_dataset_train = MNIST(root="./data", train=True, download=True)
mnist_dataset_test = MNIST(root="./data", train=False, download=True)

train_input = np.reshape(mnist_dataset_train.data.numpy(), (60000, 784))/255.
train_input_flat = train_input.flatten()
train_output = torch.nn.functional.one_hot(mnist_dataset_train.targets, num_classes=10).numpy() * (OUT_MAX - OUT_MIN) + OUT_MIN
train_output_flat = train_output.flatten()

test_input = np.reshape(mnist_dataset_test.data.numpy(), (10000, 784))/255.
test_input_flat = test_input.flatten()
test_output = torch.nn.functional.one_hot(mnist_dataset_test.targets, num_classes=10).numpy() * (OUT_MAX - OUT_MIN) + OUT_MIN
test_output_flat = test_output.flatten()

rng = np.random.default_rng()

############################
N_IN = 784
N_HIDDEN = [500]
N_OUT = 10

DT = 1.0
############################


############################
N_EPOCHS = 10

N_TRAIN = 60000
N_TEST = 10000

N_BATCH = 128

N_UPDATE_PATTERN_TRAIN = int(N_EPOCHS * N_TRAIN / N_BATCH)
N_UPDATE_PATTERN_TEST = int(N_TEST / N_BATCH)

T_SHOW_PATTERN = 300.0
T_RUN_TRAIN = N_UPDATE_PATTERN_TRAIN * T_SHOW_PATTERN
T_RUN_TEST = N_UPDATE_PATTERN_TEST * T_SHOW_PATTERN

SAMPLE_IDS_TRAIN = np.concatenate([rng.permutation(N_TRAIN) for _ in range(N_EPOCHS)])
N_SAMPLE_IDS_TRAIN = SAMPLE_IDS_TRAIN.shape[0]
UPDATE_TIMES_TRAIN = np.arange(N_UPDATE_PATTERN_TRAIN) * T_SHOW_PATTERN
N_UPDATE_TIMES_TRAIN = UPDATE_TIMES_TRAIN.shape[0]

SAMPLE_IDS_TEST = rng.permutation(N_TEST)
N_SAMPLE_IDS_TEST = SAMPLE_IDS_TEST.shape[0]
UPDATE_TIMES_TEST = np.arange(N_UPDATE_PATTERN_TEST) * T_SHOW_PATTERN
N_UPDATE_TIMES_TEST = UPDATE_TIMES_TEST.shape[0]

N_TEST_RUN = 10
TIMES_TEST_RUN = np.linspace(0., T_RUN_TRAIN, N_TEST_RUN+1)[:-1]

TIMES_RECORD_TEST = np.linspace(0., T_RUN_TEST, 15 * N_UPDATE_TIMES_TEST)
TIMES_RECORD_TRAIN = np.linspace(0., T_RUN_TRAIN, 15 * N_UPDATE_TIMES_TRAIN)

NT_SKIP_BATCH_PLAST = 2 * int(T_SHOW_PATTERN / DT)

PARAMS_TEST_RUN = [
    {
        "T": T_RUN_TEST,
        "ext_data_pop_vars": [
            (np.zeros((1, N_BATCH, N_OUT)),
                np.array([0.]),
                "neur_output_output_pop", "ga")
        ],
        "readout_neur_pop_vars": [
            #("neur_output_output_pop", "u", TIMES_RECORD_TEST),
            #("neur_output_output_pop", "vb", TIMES_RECORD_TEST),
            ("neur_output_output_pop", "va", TIMES_RECORD_TEST),
            ("neur_output_output_pop", "d_ra", TIMES_RECORD_TEST),
            ("neur_output_output_pop", "d_ra_prev", TIMES_RECORD_TEST),
            ("neur_output_output_pop", "r", TIMES_RECORD_TEST)#,
            #("neur_output_output_pop", "r_target", TIMES_RECORD_TEST)#,
            #("neur_hidden0_pyr_pop", "vb", TIMES_RECORD_TEST),
            #("neur_hidden0_pyr_pop", "va", TIMES_RECORD_TEST),
            #("neur_hidden0_pyr_pop", "r_prev", TIMES_RECORD_TEST),
            #("neur_hidden0_int_pop", "r_prev", TIMES_RECORD_TEST),
            #("neur_hidden0_int_pop", "vb", TIMES_RECORD_TEST),
            #("neur_hidden0_int_pop", "va", TIMES_RECORD_TEST)
        ]
    }
] * N_TEST_RUN

TRAIN_READOUT = [
    #("neur_output_output_pop", "u", TIMES_RECORD_TRAIN),
    #("neur_output_output_pop", "vb", TIMES_RECORD_TRAIN),
    ("neur_output_output_pop", "va", TIMES_RECORD_TRAIN),
    ("neur_output_output_pop", "d_ra", TIMES_RECORD_TRAIN)
    #("neur_output_output_pop", "r", TIMES_RECORD_TRAIN),
    #("neur_output_output_pop", "r_target", TIMES_RECORD_TRAIN),
    #("neur_hidden0_pyr_pop", "vb", TIMES_RECORD_TRAIN),
    #("neur_hidden0_pyr_pop", "va", TIMES_RECORD_TRAIN),
    #("neur_hidden0_pyr_pop", "r_prev", TIMES_RECORD_TRAIN)#,
    #("neur_hidden0_pyr_pop", "r_eff_prev", TIMES_RECORD_TRAIN),
    #("neur_hidden0_int_pop", "r_prev", TIMES_RECORD_TRAIN),
    #("neur_hidden0_int_pop", "vb", TIMES_RECORD_TRAIN),
    #("neur_hidden0_int_pop", "va", TIMES_RECORD_TRAIN)
]

cs_in_train = {
    "model": stp_src,
    "params": {
        "n_samples_set": N_TRAIN,
        "pop_size": N_IN,
        "batch_size": N_BATCH,
        "input_id_list_size": N_SAMPLE_IDS_TRAIN,
        "input_times_list_size": N_UPDATE_TIMES_TRAIN
    },
    "vars": { "idx": 0, "t_next": 0.0},
    "extra_global_params": {
        "data": train_input_flat,
        "input_id_list": SAMPLE_IDS_TRAIN,
        "input_times_list": UPDATE_TIMES_TRAIN
    }
}

cs_out_train = {
    "model": stp_src,
    "params": {
        "n_samples_set": N_TRAIN,
        "pop_size": N_OUT,
        "batch_size": N_BATCH,
        "input_id_list_size": N_SAMPLE_IDS_TRAIN,
        "input_times_list_size": N_UPDATE_TIMES_TRAIN
    },
    "vars": { "idx": 0, "t_next": 0.0},
    "extra_global_params": {
        "data": train_output_flat,
        "input_id_list": SAMPLE_IDS_TRAIN,
        "input_times_list": UPDATE_TIMES_TRAIN
    }
}

cs_in_test = {
    "model": stp_src,
    "params": {
        "n_samples_set": N_TEST,
        "pop_size": N_IN,
        "batch_size": N_BATCH,
        "input_id_list_size": N_SAMPLE_IDS_TEST,
        "input_times_list_size": N_UPDATE_TIMES_TEST
    },
    "vars": { "idx": 0, "t_next": 0.0},
    "extra_global_params": {
        "data": test_input_flat,
        "input_id_list": SAMPLE_IDS_TEST,
        "input_times_list": UPDATE_TIMES_TEST
    }
}

cs_out_test = {
    "model": stp_src,
    "params": {
        "n_samples_set": N_TEST,
        "pop_size": N_OUT,
        "batch_size": N_BATCH,
        "input_id_list_size": N_SAMPLE_IDS_TEST,
        "input_times_list_size": N_UPDATE_TIMES_TEST
    },
    "vars": { "idx": 0, "t_next": 0.0},
    "extra_global_params": {
        "data": test_output_flat,
        "input_id_list": SAMPLE_IDS_TEST,
        "input_times_list": UPDATE_TIMES_TEST
    }
}

net = Network("network", network_model,
                      N_IN, N_HIDDEN, N_OUT,  # network size
                      0,  # maximum number of input arrays for external input
                      int(T_RUN_TRAIN / DT),  # spike buffer size
                      0,  # validation spike buffer size
                      [],#"neur_output_output_pop",
                      # "neur_hidden0_pyr_pop"],  # spike buffer populations
                      [],  # validation spike buffer populations
                      n_batches=N_BATCH,
                      n_batches_val=N_BATCH,
                      cs_in_init=cs_in_train,
                      cs_out_init=cs_out_train,
                      cs_in_init_static_twin=cs_in_test,
                      cs_out_init_static_twin=cs_out_test,
                      plastic=True,
                      dt=DT
                      )

net.align_fb_weights()
net.init_self_pred_state()

'''
results = np.load(os.path.join(TASK_BASE_FOLD, "results.npz"), allow_pickle=True)

weights = results["weights"][()]
biases = results["biases"][()]

net.set_weights(weights)
net.set_biases(biases)
#'''

net.syn_pops["syn_hidden0_pyr_pop_to_output_output_pop"].pull_var_from_device("g")
w_pp = np.array(net.syn_pops["syn_hidden0_pyr_pop_to_output_output_pop"].vars["g"].view[:])
w_pp = np.reshape(w_pp, (N_HIDDEN[0], N_OUT)).T

readout_neur_arrays, readout_syn_arrays, readout_spikes, results_validation = net.run_sim(
            T_RUN_TRAIN, None, TRAIN_READOUT, None,
            t_sign_validation=TIMES_TEST_RUN,
            data_validation=PARAMS_TEST_RUN,
            NT_skip_batch_plast=NT_SKIP_BATCH_PLAST,
            force_self_pred_state=True,
            force_fb_align=True)

net.syn_pops["syn_hidden0_pyr_pop_to_output_output_pop"].pull_var_from_device("g")
w_pp_after = np.array(net.syn_pops["syn_hidden0_pyr_pop_to_output_output_pop"].vars["g"].view[:])
w_pp_after = np.reshape(w_pp_after, (N_HIDDEN[0], N_OUT)).T

#out_r = readout_neur_arrays["neur_output_output_pop_r"]
out_va = readout_neur_arrays["neur_output_output_pop_va"]
out_d_ra = readout_neur_arrays["neur_output_output_pop_d_ra"]
#pyr_r_prev = readout_neur_arrays["neur_hidden0_pyr_pop_r_prev"]
'''
out_u = readout_neur_arrays["neur_output_output_pop_u"]
out_vb = readout_neur_arrays["neur_output_output_pop_vb"]
out_va = readout_neur_arrays["neur_output_output_pop_va"]

pyr_vb = readout_neur_arrays["neur_hidden0_pyr_pop_vb"]
pyr_va = readout_neur_arrays["neur_hidden0_pyr_pop_va"]

pyr_r_eff_prev = readout_neur_arrays["neur_hidden0_pyr_pop_r_eff_prev"]
int_r_prev = readout_neur_arrays["neur_hidden0_int_pop_r_prev"]
int_vb = readout_neur_arrays["neur_hidden0_int_pop_vb"]
int_va = readout_neur_arrays["neur_hidden0_int_pop_va"]
'''

out_r_test = [results_validation[k]["neur_var_rec"]["neur_output_output_pop_r"] for k in range(N_TEST_RUN)]
out_va_test = [results_validation[k]["neur_var_rec"]["neur_output_output_pop_va"] for k in range(N_TEST_RUN)]
out_d_ra_test = [results_validation[k]["neur_var_rec"]["neur_output_output_pop_d_ra"] for k in range(N_TEST_RUN)]
out_d_ra_prev_test = [results_validation[k]["neur_var_rec"]["neur_output_output_pop_d_ra_prev"] for k in range(N_TEST_RUN)]

'''
out_u_test = [results_validation[k]["neur_var_rec"]["neur_output_output_pop_u"] for k in range(N_TEST_RUN)]
out_vb_test = [results_validation[k]["neur_var_rec"]["neur_output_output_pop_vb"] for k in range(N_TEST_RUN)]
out_va_test = [results_validation[k]["neur_var_rec"]["neur_output_output_pop_va"] for k in range(N_TEST_RUN)]
out_r_target_test = [results_validation[k]["neur_var_rec"]["neur_output_output_pop_r_target"] for k in range(N_TEST_RUN)]
pyr_vb_test = [results_validation[k]["neur_var_rec"]["neur_hidden0_pyr_pop_vb"] for k in range(N_TEST_RUN)]
pyr_va_test = [results_validation[k]["neur_var_rec"]["neur_hidden0_pyr_pop_va"] for k in range(N_TEST_RUN)]
pyr_r_prev_test = [results_validation[k]["neur_var_rec"]["neur_hidden0_pyr_pop_r_prev"] for k in range(N_TEST_RUN)]
int_r_prev_test = [results_validation[k]["neur_var_rec"]["neur_hidden0_int_pop_r_prev"] for k in range(N_TEST_RUN)]
int_vb_test = [results_validation[k]["neur_var_rec"]["neur_hidden0_int_pop_vb"] for k in range(N_TEST_RUN)]
int_va_test = [results_validation[k]["neur_var_rec"]["neur_hidden0_int_pop_va"] for k in range(N_TEST_RUN)]
'''

plt.ion()

SAMPLE_IDS_BATCHES_TEST = np.reshape(SAMPLE_IDS_TEST[:N_UPDATE_PATTERN_TEST*N_BATCH],(N_UPDATE_PATTERN_TEST,N_BATCH))

acc = []

for k in range(N_TEST_RUN):
    acc.append(calc_class_acc_interp(UPDATE_TIMES_TEST, TIMES_RECORD_TEST,
                     test_output[SAMPLE_IDS_BATCHES_TEST], out_r_test[k]))
loss = []

for k in range(N_TEST_RUN):
    loss.append(calc_loss_interp(UPDATE_TIMES_TEST, TIMES_RECORD_TEST,
                     test_output[SAMPLE_IDS_BATCHES_TEST], out_r_test[k]))

loss = np.array(loss)
acc = np.array(acc)

plt.style.use("ggplot")

epoch_ax = np.linspace(0.,N_EPOCHS, N_TEST_RUN)

fig_loss, ax_loss = plt.subplots(1,1)
ax_loss.plot(epoch_ax, loss)
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("MSE")

fig_err, ax_err = plt.subplots(1,1)
ax_err.plot(epoch_ax, (1.-acc)*100.)
ax_err.set_xlabel("Epoch")
ax_err.set_ylabel("% Test Err.")
ax_err.set_ylim(bottom=0., top=50.)

def save_data():
    np.savez(os.path.join(TASK_BASE_FOLD, "results.npz"),
            weights=net.get_weights(),
            biases=net.get_biases(),
            acc=acc,
            out_r_test=out_r_test,
            update_times_test=UPDATE_TIMES_TEST,
            times_record_test=TIMES_RECORD_TEST,
            test_output=test_output[SAMPLE_IDS_BATCHES_TEST]
            )

import pdb
pdb.set_trace()


