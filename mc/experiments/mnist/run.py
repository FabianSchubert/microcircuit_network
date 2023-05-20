import numpy as np
import pandas as pd

import os

from models import change_detection_spikes_mnist, rates_mnist

from data.mnist.dataset import input_train, output_train, input_test, output_test

data = dict(zip(["input_train", "output_train", "input_test", "output_test"],
                [input_train, output_train, input_test, output_test]))

from ..utils import train_and_test_network

BASE_FOLD = os.path.dirname(__file__)

from itertools import product

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
    "n_hidden": [800],
    "n_out": 10,
    "dt": 1.0,
    "n_runs": 2,
    "n_epochs": 35,
    "n_batch": 128,
    "t_show_pattern": 150.,
    "n_test_run": 10,
    "optimizer_params": {
        "neur_hidden0_pyr_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 0.5*1e-1/150.}
        },
        "neur_hidden0_int_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 0.5*2e-2/150.}
        },
        "neur_output_output_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 0.5*1e-2/150.}
        },
        "syn_input_input_pop_to_hidden0_pyr_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr":0.5*1e-1/150.}
        },
        "syn_hidden0_pyr_pop_to_int_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 0.5*2e-2/150}
        },
        "syn_hidden0_int_pop_to_pyr_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 0.0}
        },
        "syn_hidden0_pyr_pop_to_output_output_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 0.5*1e-2/150.}
        }
    }
}#'''

params_rnd_fb = dict(params_base) | {"force_self_pred_state": False, "force_fb_align": False}
params_fb_align = dict(params_base) | {"force_self_pred_state": True, "force_fb_align": True}

method_params = {
    "Random Feedback": params_rnd_fb,
    "Feedback Align": params_fb_align
}

models = {
    "Spike": change_detection_spikes_mnist,
    "Rate": rates_mnist
}

df_learn = pd.DataFrame(columns=["Epoch", "Sim ID", "Accuracy", "Loss", "Model", "Method"])
df_runtime = pd.DataFrame(columns=["Runtime", "Sim ID", "Model", "Method"])

for (model_name, model), (method_name, method_param) in product(models.items(), method_params.items()):
    
    epoch_ax, acc, loss, run_time = train_and_test_network(method_param, model, data)

    df_learn = pd.concat([df_learn, 
                      pd.DataFrame({
                          "Epoch": np.repeat(epoch_ax, params_base["n_runs"]),
                          "Sim ID": np.tile(np.arange(params_base["n_runs"]), params_base["n_test_run"]),
                          "Accuracy": acc.flatten(),
                          "Loss": loss.flatten(),
                          "Model": model_name,
                          "Method": method_name 
                      })], ignore_index=True)
    
    df_runtime = pd.concat(
       [df_runtime,
        pd.DataFrame(
            {"Runtime": run_time,
             "Sim ID": np.arange(params_base["n_runs"]),
             "Model": model_name,
             "Method": method_name}
        )], ignore_index=True)

df_learn.to_csv(os.path.join(BASE_FOLD, "df_learn.csv"), index=False)
df_runtime.to_csv(os.path.join(BASE_FOLD, "df_runtime.csv"), index=False)