
import sys

import numpy as np
import pandas as pd

import os

from genn_models.randman import change_detection_spikes, rates

from data.randman.dataset import randman_dataset_array

from ..utils import train_and_test_network, split_lst

from itertools import product

from tqdm import tqdm

import time

JOB_ID = int(sys.argv[2])
N_JOBS = int(sys.argv[3])
MODEL = sys.argv[1]

assert MODEL in ("spike", "rate"), "Model parameter must be either 'spike' or 'rate'"

# print(f"job id: {JOB_ID}")
# print(f"n jobs: {N_JOBS}")

D_MF = 1
N_CUTOFF = 1000
ALPHA = 1.0

SEED_MF = 42

N_SWEEP_THRESHOLD = 10
if MODEL=="spike":
    SPIKE_THRESHOLDS = np.exp(np.linspace(np.log(1e-5), np.log(1e-1), N_SWEEP_THRESHOLD))
else:
    SPIKE_THRESHOLDS = np.array([0.0])

N_SWEEP_NET_SIZE = 10
N_HIDDEN = np.linspace(10, 300, N_SWEEP_NET_SIZE).astype("int").tolist()
N_INPUT = [n + 20 - N_HIDDEN[0] for n in N_HIDDEN]
N_OUTPUT = [n + 10 - N_HIDDEN[0] for n in N_HIDDEN]

NET_SIZE_ZIP = list(zip(N_INPUT, N_HIDDEN, N_OUTPUT))

N_SAMPLES_TRAIN = 1500
N_SAMPLES_TEST = 1000

N_EPOCHS = 1500
T_SHOW_PATTERN = 150.

AX_REC_MOD_PARAMS = np.arange(N_EPOCHS) * T_SHOW_PATTERN

BASE_FOLD = os.path.dirname(__file__)

DEFAULT_ADAM_PARAMS = {
    "lr": 5e-4,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-2,
    "high": 1e6,
    "low": -1e6
}


params_base = {
    "name": f"network_job_{JOB_ID}",
    "n_in": None,
    "n_hidden": None,
    "n_out": None,
    "dt": 1.0,
    "n_runs": 1,
    "n_epochs": N_EPOCHS,
    "n_batch": 64,
    "t_show_pattern": T_SHOW_PATTERN,
    "n_test_run": 20,
    "cuda_visible_devices": True,
    "train_readout_syn": [
        ("syn_hidden0_pyr_pop_to_output_output_pop", "g", AX_REC_MOD_PARAMS),
        ("syn_output_output_pop_to_hidden0_pyr_pop", "g", AX_REC_MOD_PARAMS),
        ("syn_hidden0_pyr_pop_to_int_pop", "g", AX_REC_MOD_PARAMS),
        ("syn_hidden0_int_pop_to_pyr_pop", "g", AX_REC_MOD_PARAMS)
        ],
    "train_readout": [
        ("neur_output_output_pop", "b", AX_REC_MOD_PARAMS),
        ("neur_hidden0_int_pop", "b", AX_REC_MOD_PARAMS)
        ],
    "optimizer_params": {
        "neur_hidden0_pyr_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 5e-3}
        },
        "neur_hidden0_int_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 1e-5}
        },
        "neur_output_output_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 2e-4}
        },
        "syn_input_input_pop_to_hidden0_pyr_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 5e-3}
        },
        "syn_hidden0_pyr_pop_to_int_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 3e-4}
        },
        "syn_hidden0_int_pop_to_pyr_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 3e-4}
        },
        "syn_hidden0_pyr_pop_to_output_output_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 2e-4}
        }
    }
}

params_fb_align = {"force_self_pred_state": False,
                   "force_fb_align": False}
params_backprop = {"force_self_pred_state": True,
                   "force_fb_align": True}

method_params = {
    "Feedback Align": params_fb_align,
    "Backprop": params_backprop
}

models = {"Spike": change_detection_spikes} if MODEL=="spike" else {"Rate": rates}

df_learn = pd.DataFrame(columns=["Epoch", "Sim ID", "Accuracy",
                                 "Loss", "Model", "Method",
                                 "N Input", "N Hidden", "N Output"])
df_runtime = pd.DataFrame(columns=["Runtime", "Sim ID", "Model", "Method",
                                   "N Input", "N Hidden", "N Output"])

sweep_params = list(product(models.items(), method_params.items(), NET_SIZE_ZIP, SPIKE_THRESHOLDS))

n_sweep = len(sweep_params) # len(models) * len(method_params) * len(N_HIDDEN) * len(SPIKE_THRESHOLDS)

sweep_params_split = split_lst(sweep_params, N_JOBS)

sweep_params_instance = sweep_params_split[JOB_ID]

n_sweep_instance = len(sweep_params_instance)

for k in range(n_sweep_instance):
	size = sweep_params_instance[k][-2]
	print(f"Job ID: {JOB_ID}, {size[0]}, {size[1]}, {size[2]}")

with open(os.path.join(BASE_FOLD, "runtime_est.log"), "a") as file_log:

    t0 = time.time()

    for k_sweep, ((model_name, model), (method_name, method_param), net_size_zip, spike_th) in enumerate(sweep_params_instance):

        input_train, output_train = randman_dataset_array(net_size_zip[0], net_size_zip[2],
                                                          N_SAMPLES_TRAIN, D_MF, N_CUTOFF, ALPHA,
                                                          seed_mf=SEED_MF)
        input_test, output_test = randman_dataset_array(net_size_zip[0], net_size_zip[2],
                                                        N_SAMPLES_TEST,
                                                        D_MF, N_CUTOFF, ALPHA,
                                                        seed_mf=SEED_MF)

        data = dict(zip(["input_train", "output_train", "input_test", "output_test"],
                    [input_train, output_train, input_test, output_test]))

        _tmp_params = params_base | method_param | {"n_in": net_size_zip[0],
                                                    "n_hidden": [net_size_zip[1]],
                                                    "n_out": net_size_zip[2]}
        if MODEL=="spike":
            model.neurons.pyr.mod_dat["param_space"]["th"] = spike_th
            model.neurons.output.mod_dat["param_space"]["th"] = spike_th
            model.neurons.int.mod_dat["param_space"]["th"] = spike_th
            model.neurons.input.mod_dat["param_space"]["th"] = spike_th

        epoch_ax, acc, loss, rec_neur, rec_syn, run_time = train_and_test_network(_tmp_params,
                                                            model, data)

        w_op = rec_syn["syn_hidden0_pyr_pop_to_output_output_pop_g"]
        w_po = rec_syn["syn_output_output_pop_to_hidden0_pyr_pop_g"]
        w_po_t = w_po.swapaxes(1, 2)
        w_ip = rec_syn["syn_hidden0_pyr_pop_to_int_pop_g"]
        w_pi = rec_syn["syn_hidden0_int_pop_to_pyr_pop_g"]

        w_fb_aln = (w_op * w_po_t).sum(axis=(1,2)) / np.sqrt((w_op**2.).sum(axis=(1,2)) * (w_po_t**2.).sum(axis=(1,2)))

        w_self_pred_fwd_aln = (w_op * w_ip).sum(axis=(1,2)) / np.sqrt((w_op**2.).sum(axis=(1,2)) * (w_ip**2.).sum(axis=(1,2)))

        w_self_pred_bw_aln = (w_po * w_pi).sum(axis=(1,2)) / np.sqrt((w_po**2.).sum(axis=(1,2)) * (w_pi**2.).sum(axis=(1,2)))

        b_i = rec_neur["neur_hidden0_int_pop_b"][:,0] # only use the first batch (they are all the same value across batches anyway)
        b_o = rec_neur["neur_output_output_pop_b"][:,0] # ""

        b_self_pred_aln = (b_i * b_o).sum(axis=1) / np.sqrt((b_i**2.).sum(axis=1) * (b_o**2.).sum(axis=1))

        df_learn = pd.concat([df_learn,
                            pd.DataFrame({
                                "Epoch": np.repeat(epoch_ax,
                                                params_base["n_runs"]),
                                "Sim ID": np.tile(np.arange(params_base["n_runs"]),
                                                params_base["n_test_run"]),
                                "Accuracy": acc.flatten(),
                                "Loss": loss.flatten(),
                                "Model": model_name,
                                "Method": method_name,
                                "Spike Threshold": spike_th,
                                "N Input": net_size_zip[0],
                                "N Hidden": net_size_zip[1],
                                "N Output": net_size_zip[2]
                            })], ignore_index=True)

        df_learn = pd.concat([df_learn,
                              pd.DataFrame({
                                "Epoch": np.repeat(AX_REC_MOD_PARAMS / T_SHOW_PATTERN,
                                                    params_base["n_runs"]),
                                "Sim ID": np.tile(np.arange(params_base["n_runs"]),
                                                    AX_REC_MOD_PARAMS.shape[0]),
                                "Feedback Weight Alignment": w_fb_aln.flatten(),
                                "Self-Prediction Forward Weight Alignment": w_self_pred_fwd_aln.flatten(),
                                "Self-Prediction Backward Weight Alignment": w_self_pred_bw_aln.flatten(),
                                "Self-Prediction Bias Alignment": b_self_pred_aln.flatten(),
                                "Model": model_name,
                                "Method": method_name,
                                "Spike Threshold": spike_th,
                                "N Input": net_size_zip[0],
                                "N Hidden": net_size_zip[1],
                                "N Output": net_size_zip[2]
                            })], ignore_index=True)

        df_runtime = pd.concat(
            [df_runtime,
             pd.DataFrame(
                {"Runtime": run_time,
                 "Sim ID": np.arange(params_base["n_runs"]),
                 "Model": model_name,
                 "Method": method_name,
                 "Spike Threshold": spike_th,
                 "N Input": net_size_zip[0],
                 "N Hidden": net_size_zip[1],
                 "N Output": net_size_zip[2]}
             )], ignore_index=True)

        t1 = time.time()

        t_est_left = ((t1 - t0) / (k_sweep + 1)) * (n_sweep_instance - (k_sweep + 1))
        file_log.write(f"Job #{JOB_ID}: " + time.strftime("%H:%M:%S", time.gmtime(t_est_left)) + "\n")
        file_log.flush()

file_learn = os.path.join(BASE_FOLD, f"results_data/df_learn_{MODEL}_{JOB_ID}.csv")
file_runtime = os.path.join(BASE_FOLD, f"results_data/df_runtime_rate_{MODEL}_{JOB_ID}.csv")

df_learn.to_csv(file_learn)
df_runtime.to_csv(file_runtime)

