"""
Train the microcircuit model on the random manifold dataset
over a set of parameters and models.

To run the model on a single machine/node without slurm,
run python3 -m experiments.mc.randman.run 0 0 1 <scaling mode>
from /mc, where <scaling mode> should be "lin", "sqrt", or "const".
This defines how the size of the output layer is varied as a function
of the hidden layer size.

For usage with slurm, just call ./submit_jobs <job array size>.
"""

import numpy as np
import pandas as pd

import os

import sys

import time

from genn_models.drop_in_synapses import change_detection_spikes, rates

from data.randman.dataset import randman_dataset_array

from ...utils import split_lst
from ..utils import train_and_test_network

from itertools import product

JOB_ID = sys.argv[1]
JOB_ARRAY_ID = int(sys.argv[2])
N_JOBS = int(sys.argv[3])

BASE_FOLD = os.path.dirname(__file__)

OUTPUT_SCALING_MODE = sys.argv[4]
assert OUTPUT_SCALING_MODE in ["lin", "sqrt", "const"], \
	"scaling mode must be lin, sqrt or const"

save_path = os.path.join(BASE_FOLD, f"results_data/{OUTPUT_SCALING_MODE}")
os.makedirs(save_path, exist_ok=True)

file_learn = os.path.join(save_path, f"df_learn_{JOB_ID}.csv")
file_runtime = os.path.join(save_path, f"df_runtime_{JOB_ID}.csv")

N_RUNS = 1

###############
D_MF = 1
N_CUTOFF = 1000
ALPHA = 1.0

SEED_MF = 42

N_SWEEP_THRESHOLD = 10
N_SWEEP_NET_SIZE = 10

#SPIKE_THRESHOLDS = np.array([1e-4])

SPIKE_THRESHOLDS = np.exp(np.linspace(np.log(1e-5), np.log(1e-1),
                                      N_SWEEP_THRESHOLD))

#N_HIDDEN = [20]
#N_INPUT = [20]
#N_OUTPUT = [10]

N_HIDDEN = np.linspace(20, 300, N_SWEEP_NET_SIZE).astype("int").tolist()
N_INPUT = [n + 20 - N_HIDDEN[0] for n in N_HIDDEN]
if OUTPUT_SCALING_MODE == "lin":
    N_OUTPUT = [n + 10 - N_HIDDEN[0] for n in N_HIDDEN]
elif OUTPUT_SCALING_MODE == "sqrt":
    N_OUTPUT = [int(np.sqrt(n)) + 10 - int(np.sqrt(N_HIDDEN[0])) for n in N_HIDDEN]
else:
    N_OUTPUT = [10 for _ in N_HIDDEN]

NET_SIZE_ZIP = list(zip(N_INPUT, N_HIDDEN, N_OUTPUT))
###############

###############
N_SAMPLES_TRAIN = 1500
N_SAMPLES_TEST = 1000
###############


DEFAULT_ADAM_PARAMS = {
    "lr": 1e-3,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-3,
    "high": 1e6,
    "low": -1e6
}

LR_SCALING = 0.3

params_base = {
    "name": f"network_job_{JOB_ARRAY_ID}",
    "n_in": None,
    "n_hidden": None,
    "n_out": None,
    "dt": 0.25,
    "n_runs": 1,
    "n_epochs": 3000,
    "n_batch": 128,
    "t_show_pattern": 150.,
    "n_test_run": 10,
    "cuda_visible_devices": True,
    "optimizer_params": {
        "neur_hidden0_pyr_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": LR_SCALING * 0.5 * 5e-3}
        },
        "neur_hidden0_int_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": LR_SCALING * 0.5 * 1e-3}
        },
        "neur_output_output_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": LR_SCALING * 0.5 * 2e-4}
        },
        "syn_input_input_pop_to_hidden0_pyr_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": LR_SCALING * 0.8 * 5e-3}
        },
        "syn_hidden0_pyr_pop_to_int_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": LR_SCALING * 0.3 * 1e-3}
        },
        "syn_hidden0_int_pop_to_pyr_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": LR_SCALING * 0.0 * 1e-3}
        },
        "syn_hidden0_pyr_pop_to_output_output_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": LR_SCALING * 0.5 * 2e-4}
        }
    }
}

df_learn = pd.DataFrame(columns=["Epoch", "Sim ID", "Accuracy",
                                 "Loss", "Model", "Method",
                                 "Event Threshold",
                                 "N Input", "N Hidden", "N Output"
                                 ])
df_runtime = pd.DataFrame(columns=["Runtime", "Sim ID",
                                   "Model", "Method",
                                   "Event Threshold",
                                   "N Input", "N Hidden", "N Output"])

################ construct parameter sweep space
params_fb_align = {"force_self_pred_state": False,
                   "force_fb_align": False}
params_backprop = {"force_self_pred_state": True,
                   "force_fb_align": True}

method_params = {
    "Feedback Align": params_fb_align,
    "Backprop": params_backprop
}

models = {
    "Event": change_detection_spikes,
    "Continuous": rates
}

params_list = []

params_list.extend(list(product([list(models.items())[0]], list(method_params.items()),
                                NET_SIZE_ZIP, SPIKE_THRESHOLDS, range(N_RUNS))))

params_list.extend(list(product([list(models.items())[1]], list(method_params.items()),
                                NET_SIZE_ZIP, [0.0], range(N_RUNS))))

n_params = len(params_list)

params_split = split_lst(params_list, N_JOBS)

params_instance = params_split[JOB_ARRAY_ID]

n_params_instance = len(params_instance)

with open(os.path.join(BASE_FOLD, "runtime_est.log"), "a") as file_log:

    t0 = time.time()

    for k_sweep, ((model_name, model), (method_name, method_param), net_size_zip, spike_th, sim_id) in enumerate(params_instance):

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
        if model_name=="Event":
            model.neurons.pyr.mod_dat["param_space"]["th"] = spike_th
            model.neurons.output.mod_dat["param_space"]["th"] = spike_th
            model.neurons.int.mod_dat["param_space"]["th"] = spike_th
            model.neurons.input.mod_dat["param_space"]["th"] = spike_th

        #epoch_ax, acc, loss, rec_neur, rec_syn, run_time = train_and_test_network(_tmp_params,
        #                                                    model, data)

        epoch_ax, acc, loss, _, _, run_time = train_and_test_network(_tmp_params, model, data)

        df_learn = pd.concat([df_learn,
                              pd.DataFrame({
                                "Epoch": epoch_ax,
                                "Sim ID": sim_id,
                                "Accuracy": acc.flatten(),
                                "Loss": loss.flatten(),
                                "Model": model_name,
                                "Method": method_name,
                                "Event Threshold": spike_th,
                                "N Input": net_size_zip[0],
                                "N Hidden": net_size_zip[1],
                                "N Output": net_size_zip[2]
                              })], ignore_index=True)

        df_runtime = pd.concat(
            [df_runtime,
             pd.DataFrame(
                {"Runtime": run_time,
                 "Sim ID": sim_id,
                 "Model": model_name,
                 "Method": method_name,
                 "Event Threshold": spike_th,
                 "N Input": net_size_zip[0],
                 "N Hidden": net_size_zip[1],
                 "N Output": net_size_zip[2]}
             )], ignore_index=True)

        t1 = time.time()

        t_est_left = ((t1 - t0) / (k_sweep + 1)) * (n_params_instance - (k_sweep + 1))
        file_log.write(f"Job #{JOB_ARRAY_ID}: " + time.strftime("%H:%M:%S", time.gmtime(t_est_left)) + "\n")
        file_log.flush()

df_learn.to_csv(file_learn, index=False)
df_runtime.to_csv(file_runtime, index=False)

