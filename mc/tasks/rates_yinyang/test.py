import matplotlib.pyplot as plt

import numpy as np

import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader

from mc.network import Network

from models import rates_yinyang as network_model
from models.cs_sources import step_source_model as stp_src

from ...misc.utils import plot_spike_times, calc_loss_interp, calc_class_acc_interp

import os

TASK_BASE_FOLD = os.path.dirname(__file__)

OUT_MIN = 0.0
OUT_MAX = 1.0

from data.yinyang.dataset import YinYangDataset


dataset_train = YinYangDataset(size=5000, seed=42)
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


############################
N_EPOCHS = 1300

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
            ("neur_input_input_pop", "r", TIMES_RECORD_TEST),
            #("neur_output_output_pop", "u", TIMES_RECORD_TEST),
            ("neur_output_output_pop", "vb", TIMES_RECORD_TEST),
            ("neur_output_output_pop", "va", TIMES_RECORD_TEST),
            ("neur_output_output_pop", "d_ra", TIMES_RECORD_TEST),
            ("neur_output_output_pop", "r", TIMES_RECORD_TEST)
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
    ("neur_input_input_pop", "r", TIMES_RECORD_TRAIN),
    #("neur_output_output_pop", "u", TIMES_RECORD_TRAIN),
    ("neur_output_output_pop", "vb", TIMES_RECORD_TRAIN),
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

TRAIN_READOUT_SYN = [
    ("syn_hidden0_pyr_pop_to_output_output_pop", "g", TIMES_RECORD_TRAIN_SYN),
    ("syn_hidden0_pyr_pop_to_int_pop", "g", TIMES_RECORD_TRAIN_SYN)
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
                      int(T_SHOW_PATTERN * 100.0 / DT),  # spike buffer size
                      0,  # validation spike buffer size
                      ["neur_hidden0_pyr_pop"],#"neur_output_output_pop",
                      # "neur_hidden0_pyr_pop"],  # spike buffer populations
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


'''weights = np.load("./data/yinyang/yinyang_weights.npz")

net.set_weights(
    {
        "syn_input_input_pop_to_hidden0_pyr_pop": weights["w_10"].flatten(),
        "syn_hidden0_pyr_pop_to_hidden1_pyr_pop": weights["w_21"].flatten(),
        "syn_hidden1_pyr_pop_to_output_output_pop": weights["w_32"].flatten(),
    }
)#'''

'''
results = np.load(os.path.join(TASK_BASE_FOLD, "results.npz"), allow_pickle=True)

weights = results["weights"][()]
biases = results["biases"][()]

net.set_weights(weights)
net.set_biases(biases)
#'''

#net.align_fb_weights()
net.init_self_pred_state()

readout_neur_arrays, readout_syn_arrays, readout_spikes, results_validation = net.run_sim(
            T_RUN_TRAIN, None, TRAIN_READOUT, TRAIN_READOUT_SYN,
            t_sign_validation=TIMES_TEST_RUN,
            data_validation=PARAMS_TEST_RUN,
            NT_skip_batch_plast=NT_SKIP_BATCH_PLAST,
            force_self_pred_state=False,
            force_fb_align=False)

#out_r = readout_neur_arrays["neur_output_output_pop_r"]
inp_r = readout_neur_arrays["neur_input_input_pop_r"]
out_vb = readout_neur_arrays["neur_output_output_pop_vb"]
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

inp_r_test = [results_validation[k]["neur_var_rec"]["neur_input_input_pop_r"] for k in range(N_TEST_RUN)]
out_r_test = [results_validation[k]["neur_var_rec"]["neur_output_output_pop_r"] for k in range(N_TEST_RUN)]
out_vb_test = [results_validation[k]["neur_var_rec"]["neur_output_output_pop_vb"] for k in range(N_TEST_RUN)]
out_va_test = [results_validation[k]["neur_var_rec"]["neur_output_output_pop_va"] for k in range(N_TEST_RUN)]
out_d_ra_test = [results_validation[k]["neur_var_rec"]["neur_output_output_pop_d_ra"] for k in range(N_TEST_RUN)]

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

w_pp = w_pp = readout_syn_arrays["syn_hidden0_pyr_pop_to_output_output_pop_g"]
w_ip = readout_syn_arrays["syn_hidden0_pyr_pop_to_int_pop_g"]

weight_align = (w_ip * w_pp).sum(axis=-1)/(np.linalg.norm(w_ip,axis=-1)*np.linalg.norm(w_pp,axis=-1))

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

fig_loss.savefig(os.path.join(TASK_BASE_FOLD, "plots/loss.png"))

fig_err, ax_err = plt.subplots(1,1)
ax_err.plot(epoch_ax, (1.-acc)*100.)
ax_err.set_xlabel("Epoch")
ax_err.set_ylabel("% Test Err.")
#ax_err.set_ylim(bottom=0., top=50.)

fig_err.savefig(os.path.join(TASK_BASE_FOLD, "plots/err.png"))

fig_align, ax_align = plt.subplots(1,1)
ax_align.plot(TIMES_RECORD_TRAIN_SYN, weight_align)

fig_align.savefig(os.path.join(TASK_BASE_FOLD, "plots/align.png"))

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


