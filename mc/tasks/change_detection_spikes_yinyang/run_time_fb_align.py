import matplotlib.pyplot as plt

import numpy as np

import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader

from mc.network import Network

from models import change_detection_spikes_yinyang as network_model
from models.cs_sources import step_source_model as stp_src

from ...misc.utils import plot_spike_times, calc_loss_interp, calc_class_acc_interp

import os

TASK_BASE_FOLD = os.path.dirname(__file__)

OUT_MIN = 0.0
OUT_MAX = 1.0

from data.yinyang.dataset import YinYangDataset

import time

from tqdm import tqdm

dataset_train = YinYangDataset(size=5000, seed=40)
dataset_test = YinYangDataset(size=1000, seed=41)

#yinyang_data = np.load("./data/yinyang/yinyang.npz")

#train_input = yinyang_data["train_input"]
train_input = np.array(dataset_train._YinYangDataset__vals)
train_input_flat = train_input.flatten()

train_output = one_hot(torch.tensor(np.array(dataset_train._YinYangDataset__cs)), num_classes=3).detach().numpy() * (OUT_MAX - OUT_MIN) + OUT_MIN
#train_output = yinyang_data["train_output"] * (OUT_MAX - OUT_MIN) + OUT_MIN
train_output_flat = train_output.flatten()

#test_input = yinyang_data["test_input"]
test_input = np.array(dataset_test._YinYangDataset__vals)
test_input_flat = test_input.flatten()

#test_output = yinyang_data["test_output"] * (OUT_MAX - OUT_MIN) + OUT_MIN
test_output = one_hot(torch.tensor(np.array(dataset_test._YinYangDataset__cs)), num_classes=3).detach().numpy() * (OUT_MAX - OUT_MIN) + OUT_MIN
test_output_flat = test_output.flatten()

rng = np.random.default_rng()

############################
N_IN = train_input.shape[1]
N_HIDDEN = [150]
N_OUT = train_output.shape[1]

DT = 1.0
############################

N_SAMPLES_TIMING = 10

############################
N_EPOCHS = 1500

N_TRAIN = train_input.shape[0]
N_TEST = test_input.shape[0]

N_BATCH = 128

N_UPDATE_PATTERN_TRAIN = int(N_EPOCHS * N_TRAIN / N_BATCH)
N_UPDATE_PATTERN_TEST = int(N_TEST / N_BATCH)

T_SHOW_PATTERN = 150.0
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

N_TEST_RUN = 20
TIMES_TEST_RUN = np.linspace(0., T_RUN_TRAIN, N_TEST_RUN+1)[:-1]

TIMES_RECORD_TEST = np.linspace(0., T_RUN_TEST, 30 * N_UPDATE_TIMES_TEST)
TIMES_RECORD_TRAIN = np.linspace(0., T_RUN_TRAIN, int(0.25 * N_UPDATE_TIMES_TRAIN))
TIMES_RECORD_TRAIN_SYN = np.linspace(0., T_RUN_TRAIN, int(0.1 * N_UPDATE_TIMES_TRAIN))

NT_SKIP_BATCH_PLAST = int(T_SHOW_PATTERN / DT)

DEFAULT_ADAM_PARAMS = {
    "lr": 1e-3,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-7,
    "high": 1e6,
    "low": -1e6
}

OPTIMIZER_PARAMS = {
    "neur_hidden0_pyr_pop": {
        "optimizer": "adam",
        "params": DEFAULT_ADAM_PARAMS | {"lr": 1e-1/T_SHOW_PATTERN}
    },
    "neur_hidden0_int_pop": {
        "optimizer": "adam",
        "params": DEFAULT_ADAM_PARAMS | {"lr": 2e-2/T_SHOW_PATTERN}
    },
    "neur_output_output_pop": {
        "optimizer": "adam",
        "params": DEFAULT_ADAM_PARAMS | {"lr": 1e-2/T_SHOW_PATTERN}
    },
    "syn_input_input_pop_to_hidden0_pyr_pop": {
        "optimizer": "adam",
        "params": DEFAULT_ADAM_PARAMS | {"lr": 1e-1/T_SHOW_PATTERN}
    },
    "syn_hidden0_pyr_pop_to_int_pop": {
        "optimizer": "adam",
        "params": DEFAULT_ADAM_PARAMS | {"lr": 2e-2/T_SHOW_PATTERN}
    },
    "syn_hidden0_int_pop_to_pyr_pop": {
        "optimizer": "adam",
        "params": DEFAULT_ADAM_PARAMS | {"lr": 0.0}
    },
    "syn_hidden0_pyr_pop_to_output_output_pop": {
        "optimizer": "adam",
        "params": DEFAULT_ADAM_PARAMS | {"lr": 1e-2/T_SHOW_PATTERN}
    }
}

PARAMS_TEST_RUN = [
    {
        "T": T_RUN_TEST,
        "ext_data_pop_vars": [
            (np.zeros((1, N_BATCH, N_OUT)),
                np.array([0.]),
                "neur_output_output_pop", "ga")
        ],
        "readout_neur_pop_vars": [
            ("neur_output_output_pop", "r", TIMES_RECORD_TEST)
        ]
    }
] * N_TEST_RUN

TRAIN_READOUT = []

TRAIN_READOUT_SYN = []

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

acc = np.ndarray((N_TEST_RUN, N_SAMPLES_TIMING))
loss = np.ndarray((N_TEST_RUN, N_SAMPLES_TIMING))
run_time = np.ndarray((N_SAMPLES_TIMING))

net = Network("network", network_model,
                N_IN, N_HIDDEN, N_OUT,  # network size
                0,  # maximum number of input arrays for external input
                0,  # spike buffer size
                0,  # validation spike buffer size
                [],  # spike buffer populations
                [],  # validation spike buffer populations
                n_batches=N_BATCH,
                n_batches_val=N_BATCH,
                cs_in_init=cs_in_train,
                cs_out_init=cs_out_train,
                cs_in_init_static_twin=cs_in_test,
                cs_out_init_static_twin=cs_out_test,
                plastic=True,
                dt=DT,
                optimizer_params=OPTIMIZER_PARAMS
                )

for run_id in tqdm(range(N_SAMPLES_TIMING)):

    net.genn_model.reinitialise()

    net.init_self_pred_state()

    t0 = time.time()

    readout_neur_arrays, readout_syn_arrays, readout_spikes, results_validation = net.run_sim(
                T_RUN_TRAIN, None, TRAIN_READOUT, TRAIN_READOUT_SYN,
                t_sign_validation=TIMES_TEST_RUN,
                data_validation=PARAMS_TEST_RUN,
                NT_skip_batch_plast=NT_SKIP_BATCH_PLAST,
                force_self_pred_state=True,
                force_fb_align=True,show_progress=False)

    t1 = time.time()

    out_r_test = [results_validation[k]["neur_var_rec"]["neur_output_output_pop_r"] for k in range(N_TEST_RUN)]

    SAMPLE_IDS_BATCHES_TEST = np.reshape(SAMPLE_IDS_TEST[:N_UPDATE_PATTERN_TEST*N_BATCH],(N_UPDATE_PATTERN_TEST,N_BATCH))

    _acc = []

    for k in range(N_TEST_RUN):
        _acc.append(calc_class_acc_interp(UPDATE_TIMES_TEST, TIMES_RECORD_TEST,
                        test_output[SAMPLE_IDS_BATCHES_TEST], out_r_test[k]))
    _loss = []

    for k in range(N_TEST_RUN):
        _loss.append(calc_loss_interp(UPDATE_TIMES_TEST, TIMES_RECORD_TEST,
                        test_output[SAMPLE_IDS_BATCHES_TEST], out_r_test[k]))

    _acc = np.array(_acc)
    _loss = np.array(_loss)

    acc[:,run_id] = _acc
    loss[:,run_id] = _loss
    run_time[run_id] = t1 - t0

epoch_ax = np.linspace(0.,N_EPOCHS, N_TEST_RUN)

def save_data():
    np.savez(os.path.join(TASK_BASE_FOLD, "results_data/results_run_time_fb_align.npz"),
            acc=acc,
            loss=loss,
            run_time=run_time,
            epoch_ax=epoch_ax)

save_data()


