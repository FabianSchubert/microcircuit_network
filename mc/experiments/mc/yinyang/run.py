"""
Train the microcircuit model on the yinyang dataset
over a set of parameters and models.

To run the model on a single machine/node without slurm,
run python3 -m experiments.mc.yinyang.run 0 0 1.

For usage with slurm, just call ./submit_jobs <job array size>.
"""

import numpy as np
import pandas as pd

import os
import sys
import time

from genn_models.drop_in_synapses import change_detection_spikes, rates

from data.yinyang.dataset import yinyang_dataset_array

from ...utils import split_lst
from ..utils import train_and_test_network

from itertools import product

input_train, output_train = yinyang_dataset_array(size=10000, seed=40)
input_test, output_test = yinyang_dataset_array(size=1000, seed=40)

data = dict(zip(["input_train", "output_train", "input_test", "output_test"],
                [input_train, output_train, input_test, output_test]))

JOB_ID = sys.argv[1]
JOB_ARRAY_ID = int(sys.argv[2])
N_JOBS = int(sys.argv[3])

BASE_FOLD = os.path.dirname(__file__)

save_path = os.path.join(BASE_FOLD, f"results_data")
os.makedirs(save_path, exist_ok=True)

file_learn = os.path.join(save_path, f"df_learn_{JOB_ID}.csv")
file_runtime = os.path.join(save_path, f"df_runtime_{JOB_ID}.csv")

N_RUNS = 1

DEFAULT_ADAM_PARAMS = {
    "lr": 5e-4,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-2,
    "high": 1e6,
    "low": -1e6
}

params_base = {
    "n_in": 4,
    "n_hidden": [30],
    "n_out": 3,
    "dt": 1.0,
    "n_runs": 1,
    "n_epochs": 500,
    "n_batch": 128,
    "t_show_pattern": 150.,
    "n_test_run": 20,
    "optimizer_params": {
        "neur_hidden0_pyr_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 5e-3}
        },
        "neur_hidden0_int_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 1e-3}
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
            "params": DEFAULT_ADAM_PARAMS | {"lr": 1e-3}
        },
        "syn_hidden0_int_pop_to_pyr_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 0.0}
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

models = {
    "Event": change_detection_spikes,
    "Continuous": rates
}

params_list = list(product(list(models.items()),
                           list(method_params.items()),
                           range(N_RUNS)))

n_params = len(params_list)
params_split = split_lst(params_list, N_JOBS)
params_instance = params_split[JOB_ARRAY_ID]
n_params_instance = len(params_instance)

df_learn = pd.DataFrame(columns=["Epoch", "Sim ID", "Accuracy",
                                 "Loss", "Model", "Method"])
df_runtime = pd.DataFrame(columns=["Runtime", "Sim ID", "Model", "Method"])


with open(os.path.join(BASE_FOLD, "runtime_est.log"), "a") as file_log:

    t0 = time.time()

    for k_sweep, ((model_name, model), (method_name, method_param), sim_id) in enumerate(params_instance):

        _tmp_params = params_base | method_param

        epoch_ax, acc, loss, _, _, run_time = train_and_test_network(_tmp_params, model, data)

        df_learn = pd.concat([
            df_learn,
            pd.DataFrame({
                "Epoch": epoch_ax,
                "Sim ID": sim_id,
                "Accuracy": acc.flatten(),
                "Loss": loss.flatten(),
                "Model": model_name,
                "Method": method_name}
            )], ignore_index=True)

        df_runtime = pd.concat([
            df_runtime,
            pd.DataFrame(
                {"Runtime": run_time,
                 "Sim ID": sim_id,
                 "Model": model_name,
                 "Method": method_name}
             )], ignore_index=True)

        t1 = time.time()

        t_est_left = ((t1 - t0) / (k_sweep + 1)) * (n_params_instance - (k_sweep + 1))
        file_log.write(f"Job #{JOB_ARRAY_ID}: " + time.strftime("%H:%M:%S", time.gmtime(t_est_left)) + "\n")
        file_log.flush()

df_learn.to_csv(file_learn, index=False)
df_runtime.to_csv(file_runtime, index=False)
