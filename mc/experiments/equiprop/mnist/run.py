"""
Train the microcircuit model on the MNIST dataset
over a set of parameters and models.

To run the model on a single machine/node without slurm,
run python3 -m experiments.mc.mnist.run 0 0 1 from
/mc.

For usage with slurm, just call ./submit_jobs <job array size>.
"""

import numpy as np
import pandas as pd

import os

import sys

import time

from genn_models.drop_in_synapses_equiprop import change_detection_spikes, rates

from data.mnist.dataset import (input_train, output_train,
                                input_test, output_test)

output_train = output_train * 0.8 + 0.1
output_test = output_test * 0.8 + 0.1

from ...utils import split_lst
from ..utils import train_and_test_network

from itertools import product

JOB_ID = sys.argv[1]
JOB_ARRAY_ID = int(sys.argv[2])
N_JOBS = int(sys.argv[3])

BASE_FOLD = os.path.dirname(__file__)

save_path = os.path.join(BASE_FOLD, f"results_data")
os.makedirs(save_path, exist_ok=True)

file_learn = os.path.join(save_path, f"df_learn_{JOB_ID}.csv")
file_runtime = os.path.join(save_path, f"df_runtime_{JOB_ID}.csv")

N_RUNS = 1

data = dict(zip(["input_train", "output_train", "input_test", "output_test"],
                [input_train, output_train, input_test, output_test]))


BASE_FOLD = os.path.dirname(__file__)


DEFAULT_ADAM_PARAMS = {
    "lr": 1e-3,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-3,
    "high": 1e6,
    "low": -1e6
}

N_EPOCHS = 2
READOUT_TIMES = np.linspace(0, N_EPOCHS * 60000 * 150 * 2./128., 1000)
#READOUT_TIMES = np.linspace(0., 150*2.*25, 25*20)

params_base = {
    "n_in": 784,
    "n_hidden": [200],
    "n_out": 10,
    "dt": 0.25,
    "n_runs": 1,  # this should be set to 1 if multiple jobs are used. Instead, set N_RUNS above.
    "n_epochs": N_EPOCHS,
    "n_batch": 128,
    "t_show_pattern": 150.,
    "n_test_run": 10,
    "optimizer_params": {
        "syn_input_input_pop_to_hidden0_hidden_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 0e-3}
        },
        "syn_hidden0_hidden_pop_to_output_output_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 2e-4}
        },
        "syn_output_output_pop_to_hidden0_hidden_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 0e-5}
        }
    },
    "train_readout": [
        ("neur_output_output_pop", "u", READOUT_TIMES),
        ("neur_hidden0_hidden_pop", "u", READOUT_TIMES),
        ("neur_hidden0_hidden_pop", "r", READOUT_TIMES),
        ("neur_hidden0_hidden_pop", "targ_mode", READOUT_TIMES),
        ("neur_input_input_pop", "r", READOUT_TIMES),
        ("neur_output_output_pop", "r_targ", READOUT_TIMES),
        ("neur_output_output_pop", "r", READOUT_TIMES),
        ("neur_output_output_pop", "targ_mode", READOUT_TIMES)
    ],
    "train_readout_syn": [
        ("syn_input_input_pop_to_hidden0_hidden_pop", "g", READOUT_TIMES),
        ("syn_hidden0_hidden_pop_to_output_output_pop", "g", READOUT_TIMES)
    ]
}

models = {
    "Spike": change_detection_spikes#,
    #"Rate": rates
}

df_learn = pd.DataFrame(columns=["Epoch", "Sim ID", "Accuracy",
                                 "Loss", "Model"])
df_runtime = pd.DataFrame(columns=["Runtime", "Sim ID",
                                   "Model"])

params_list = list(product(models.items(), range(N_RUNS)))

n_params = len(params_list)

params_split = split_lst(params_list, N_JOBS)

params_instance = params_split[JOB_ARRAY_ID]

n_params_instance = len(params_instance)

with open(os.path.join(BASE_FOLD, "runtime_est.log"), "a") as file_log:

    t0 = time.time()

    for k_sweep, ((model_name, model), sim_id) in enumerate(params_instance):

        epoch_ax, acc, loss, neur_readout, syn_readout, run_time = train_and_test_network(params_base, model, data, show_progress=True)

        df_learn = pd.concat([df_learn,
                              pd.DataFrame({
                                "Epoch": epoch_ax,
                                "Sim ID": sim_id,
                                "Accuracy": acc.flatten(),
                                "Loss": loss.flatten(),
                                "Model": model_name
                              })], ignore_index=True)

        df_runtime = pd.concat(
           [df_runtime,
            pd.DataFrame(
                {"Runtime": run_time,
                 "Sim ID": sim_id,
                 "Model": model_name}
            )], ignore_index=True)

        t1 = time.time()

        t_est_left = ((t1 - t0) / (k_sweep + 1)) * (n_params_instance - (k_sweep + 1))
        file_log.write(f"Job #{JOB_ARRAY_ID}: " + time.strftime("%H:%M:%S", time.gmtime(t_est_left)) + "\n")
        file_log.flush()

df_learn.to_csv(file_learn, index=False)
df_runtime.to_csv(file_runtime, index=False)
