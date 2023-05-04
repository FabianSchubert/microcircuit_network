import matplotlib.pyplot as plt

import numpy as np

from mc.network import Network

from models import change_detection_spikes as network_model
from models.cs_sources import step_source_model as stp_src

from ..utils import plot_spike_times, calc_loss_interp, calc_class_acc_interp

import os

TASK_BASE_FOLD = os.path.dirname(__file__)

OUT_MIN = 0.1
OUT_MAX = 0.9

yinyang_data = np.load("./data/yinyang/yinyang.npz")

train_input = yinyang_data["train_input"]
train_input_flat = train_input.flatten()
train_output = yinyang_data["train_output"] * (OUT_MAX - OUT_MIN) + OUT_MIN
train_output_flat = train_output.flatten()

test_input = yinyang_data["test_input"]
test_input_flat = test_input.flatten()
test_output = yinyang_data["test_output"] * (OUT_MAX - OUT_MIN) + OUT_MIN
test_output_flat = test_output.flatten()

rng = np.random.default_rng()

############################
N_IN = train_input.shape[1]
N_HIDDEN = [100]
N_OUT = train_output.shape[1]

DT = 1.0
############################




############################
N_EPOCHS = 1500

N_TRAIN = train_input.shape[0]
N_TEST = test_input.shape[0]

N_BATCH = 256

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

NT_SKIP_BATCH_PLAST = int(T_SHOW_PATTERN / DT)

PARAMS_TEST_RUN = [
    {
        "T": T_RUN_TEST,
        "ext_data_pop_vars": [
            (np.zeros((1, N_BATCH, N_OUT)),
                np.array([0.]),
                "neur_output_output_pop", "ga")
        ],
        "readout_neur_pop_vars": [
            ("neur_input_input_pop", "r_prev", TIMES_RECORD_TEST),
            #("neur_output_output_pop", "u", TIMES_RECORD_TEST),
            ("neur_output_output_pop", "vb", TIMES_RECORD_TEST),
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
    ("neur_input_input_pop", "r_prev", TIMES_RECORD_TRAIN),
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

#net.align_fb_weights()
#net.init_self_pred_state()

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

readout_neur_arrays, readout_syn_arrays, readout_spikes, results_validation = net.run_sim(
            T_RUN_TRAIN, None, TRAIN_READOUT, None,
            t_sign_validation=TIMES_TEST_RUN,
            data_validation=PARAMS_TEST_RUN,
            NT_skip_batch_plast=NT_SKIP_BATCH_PLAST,
            force_self_pred_state=True,
            force_fb_align=True)

#out_r = readout_neur_arrays["neur_output_output_pop_r"]
inp_r_prev = readout_neur_arrays["neur_input_input_pop_r_prev"]
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

inp_r_prev_test = [results_validation[k]["neur_var_rec"]["neur_input_input_pop_r_prev"] for k in range(N_TEST_RUN)]
out_r_test = [results_validation[k]["neur_var_rec"]["neur_output_output_pop_r"] for k in range(N_TEST_RUN)]
out_vb_test = [results_validation[k]["neur_var_rec"]["neur_output_output_pop_vb"] for k in range(N_TEST_RUN)]
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


