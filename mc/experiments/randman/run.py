import numpy as np
import pandas as pd

import os

from models.randman import change_detection_spikes, rates

from data.randman.dataset import randman_dataset_array

from ..utils import train_and_test_network

from itertools import product

from tqdm import tqdm

D_EMBEDDING = 20
D_MF = 1
N_CLASSES = 10
N_CUTOFF = 1000
ALPHA = 1.0

SEED_MF = 42

N_HIDDEN = np.linspace(100, 500, 2).astype("int").tolist()

SPIKE_THRESHOLDS = np.exp(np.linspace(np.log(1e-5), np.log(1e-1), 3))
                   #np.append(10.**np.arange(-10, -2),
                   #          np.exp(np.linspace(np.log(1e-2), np.log(5e-1), 15)))

N_SAMPLES_TRAIN = 1500
N_SAMPLES_TEST = 1000

N_EPOCHS = 15
T_SHOW_PATTERN = 150.

AX_REC_MOD_PARAMS = np.arange(N_EPOCHS) * T_SHOW_PATTERN

input_train, output_train = randman_dataset_array(D_EMBEDDING, N_CLASSES,
                                                  N_SAMPLES_TRAIN, D_MF, N_CUTOFF, ALPHA,
                                                  seed_mf=SEED_MF)
input_test, output_test = randman_dataset_array(D_EMBEDDING, N_CLASSES, N_SAMPLES_TEST,
                                                D_MF, N_CUTOFF, ALPHA, seed_mf=SEED_MF)

data = dict(zip(["input_train", "output_train", "input_test", "output_test"],
                [input_train, output_train, input_test, output_test]))

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
    "n_in": D_EMBEDDING,
    "n_hidden": None,
    "n_out": N_CLASSES,
    "dt": 1.0,
    "n_runs": 1,
    "n_epochs": N_EPOCHS,
    "n_batch": 64,
    "t_show_pattern": T_SHOW_PATTERN,
    "n_test_run": 20,
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
    "Feedback Align": params_fb_align#,
    #"Backprop": params_backprop
}

models = {
    "Spike": change_detection_spikes #,
    # "Rate": rates
}

df_learn = pd.DataFrame(columns=["Epoch", "Sim ID", "Accuracy",
                                 "Loss", "Model", "Method",
                                 "Spike Threshold", "N Hidden"])
df_runtime = pd.DataFrame(columns=["Runtime", "Sim ID", "Model", "Method",
                                   "Spike Threshold", "N Hidden"])

sweep_params = product(models.items(), method_params.items(), N_HIDDEN, SPIKE_THRESHOLDS)

for (model_name, model), (method_name, method_param), n_hidden, spike_th in tqdm(sweep_params):

    _tmp_params = params_base | method_param | {"n_hidden": [n_hidden]}

    model.neurons.pyr.mod_dat["param_space"]["th"] = spike_th
    model.neurons.output.mod_dat["param_space"]["th"] = spike_th
    model.neurons.int.mod_dat["param_space"]["th"] = spike_th
    model.neurons.input.mod_dat["param_space"]["th"] = spike_th

    epoch_ax, acc, loss, rec_neur, rec_syn, run_time = train_and_test_network(_tmp_params,
                                                           model, data)

    w_op = rec_syn["syn_hidden0_pyr_pop_to_output_output_pop_g"]
    w_po = rec_syn["syn_output_output_pop_to_hidden0_pyr_pop_g"]
    w_po_t = w_po.swapaxes(1,2)
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
                            "N Hidden": n_hidden
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
                              "N Hidden": n_hidden
                          })], ignore_index=True)

    df_runtime = pd.concat(
       [df_runtime,
        pd.DataFrame(
            {"Runtime": run_time,
             "Sim ID": np.arange(params_base["n_runs"]),
             "Model": model_name,
             "Method": method_name,
             "Spike Threshold": spike_th,
             "N Hidden": n_hidden}
        )], ignore_index=True)

df_learn.to_csv(os.path.join(BASE_FOLD, "df_learn.csv"), index=False)
df_runtime.to_csv(os.path.join(BASE_FOLD, "df_runtime.csv"), index=False)
