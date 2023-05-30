import numpy as np
import pandas as pd

import os

from models.mnist_threshold_var import change_detection_spikes

from data.mnist.dataset import (input_train, output_train,
                                input_test, output_test)

from ..utils import train_and_test_network

from itertools import product

BASE_FOLD = os.path.dirname(__file__)

N_TRAIN = 10000
N_TEST = 1000

# N_TH = 10
# TH_MIN = 1e-10
# TH_MAX = 0.5

# thresholds linearly spaced in log space
SPIKE_THRESHOLDS = np.append(10.**np.arange(-10, -2),
                             np.exp(np.linspace(np.log(1e-2), np.log(5e-1), 15)))
# SPIKE_THRESHOLDS = np.exp(np.linspace(np.log(TH_MIN), np.log(TH_MAX), N_TH))
# SPIKE_THRESHOLDS = np.append(0., SPIKE_THRESHOLDS)

DEFAULT_ADAM_PARAMS = {
    "lr": 1e-3,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-3,
    "high": 1e6,
    "low": -1e6
}

params_base = {
    "n_in": 784,
    "n_hidden": [300],
    "n_out": 10,
    "dt": 1.0,
    "n_runs": 5,
    "n_epochs": 50,
    "n_batch": 128,
    "t_show_pattern": 150.,
    "n_test_run": 10,
    "optimizer_params": {
        "neur_hidden0_pyr_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 0.5 * 5e-3}
        },
        "neur_hidden0_int_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 0.5 * 1e-3}
        },
        "neur_output_output_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 0.5 * 2e-4}
        },
        "syn_input_input_pop_to_hidden0_pyr_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 0.3 * 5e-3}
        },
        "syn_hidden0_pyr_pop_to_int_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 0.3 * 1e-3}
        },
        "syn_hidden0_int_pop_to_pyr_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 0.0}
        },
        "syn_hidden0_pyr_pop_to_output_output_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 0.3 * 2e-4}
        }
    }
}  # '''

params_rnd_fb = {"force_self_pred_state": False,
                 "force_fb_align": False}
params_fb_align = {"force_self_pred_state": True,
                   "force_fb_align": True}

method_params = {
    #"Random Feedback": params_rnd_fb,
    "Feedback Align": params_fb_align
}

models = {
    "Spike": change_detection_spikes  # ,
    # "Rate": rates
}

df_learn = pd.DataFrame(columns=["Epoch", "Sim ID", "Accuracy",
                                 "Loss", "Model", "Method"])
df_runtime = pd.DataFrame(columns=["Runtime", "Sim ID",
                                   "Model", "Method"])

sweep_params = product(models.items(), method_params.items(), SPIKE_THRESHOLDS)

data = dict(zip(["input_train", "output_train",
                 "input_test", "output_test"],
                [input_train, output_train, input_test, output_test]))

for (model_name, model), (method_name, method_param), spike_th in sweep_params:

    _tmp_method = dict(params_base) | method_param

    model.neurons.pyr.mod_dat["param_space"]["th"] = spike_th
    model.neurons.output.mod_dat["param_space"]["th"] = spike_th
    model.neurons.int.mod_dat["param_space"]["th"] = spike_th
    model.neurons.input.mod_dat["param_space"]["th"] = spike_th

    epoch_ax, acc, loss, run_time = train_and_test_network(_tmp_method,
                                                           model, data)

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
                            "Spike Threshold": spike_th
                          })], ignore_index=True)

    df_runtime = pd.concat(
       [df_runtime,
        pd.DataFrame(
            {"Runtime": run_time,
             "Sim ID": np.arange(params_base["n_runs"]),
             "Model": model_name,
             "Method": method_name,
             "Spike Threshold": spike_th}
        )], ignore_index=True)

df_learn.to_csv(os.path.join(BASE_FOLD, "df_learn.csv"), index=False)
df_runtime.to_csv(os.path.join(BASE_FOLD, "df_runtime.csv"), index=False)
