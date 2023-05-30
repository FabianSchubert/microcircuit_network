import numpy as np
import pandas as pd

import os

from models.yinyang_mult import change_detection_spikes, rates

from data.yinyang.dataset import yinyang_dataset_array

from ..utils import train_and_test_network

from itertools import product

BASE_FOLD = os.path.dirname(__file__)

N_TRAIN = 10000
N_TEST = 1000

N_YINYANG_INPUT = np.arange(1, 8).tolist()

N_HIDDEN_BASE = 30

DEFAULT_ADAM_PARAMS = {
    "lr": 1e-3,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-3,
    "high": 1e6,
    "low": -1e6
}

params_base = {
    "n_in": None,
    "n_hidden": None,
    "n_out": None,
    "dt": 1.0,
    "n_runs": 5,
    "n_epochs": 1000,
    "n_batch": 128,
    "t_show_pattern": 150.,
    "n_test_run": 10,
    "optimizer_params": {
        "neur_hidden0_pyr_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 1.0 * 5e-3}
        },
        "neur_hidden0_int_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 1.0 * 1e-3}
        },
        "neur_output_output_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 1.0 * 2e-4}
        },
        "syn_input_input_pop_to_hidden0_pyr_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 1.0 * 5e-3}
        },
        "syn_hidden0_pyr_pop_to_int_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 1.0 * 1e-3}
        },
        "syn_hidden0_int_pop_to_pyr_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 0.0}
        },
        "syn_hidden0_pyr_pop_to_output_output_pop": {
            "optimizer": "adam",
            "params": DEFAULT_ADAM_PARAMS | {"lr": 1.0 * 2e-4}
        }
    }
}  #'''

params_rnd_fb = {"force_self_pred_state": False,
                 "force_fb_align": False}
params_fb_align = {"force_self_pred_state": True,
                   "force_fb_align": True}

method_params = {
    #"Random Feedback": params_rnd_fb,
    "Feedback Align": params_fb_align
}

models = {
    "Spike": change_detection_spikes,
    "Rate": rates
}

df_learn = pd.DataFrame(columns=["Epoch", "Sim ID",# "Accuracy",
                                 "Loss", "Model", "Method", "Number Input Datasets"])
df_runtime = pd.DataFrame(columns=["Runtime", "Sim ID",
                                   "Model", "Method", "Number Input Datasets"])

for (model_name, model), (method_name, method_param), n_yinyang_input in product(models.items(), method_params.items(), N_YINYANG_INPUT):

    input_train = np.ndarray((N_TRAIN, 0))
    # output_train = np.zeros((N_TRAIN, 3))
    output_train = np.ndarray((N_TRAIN, 0))

    input_test = np.ndarray((N_TEST, 0))
    # output_test = np.zeros((N_TEST, 3))
    output_test = np.ndarray((N_TEST, 0))

    for k in range(n_yinyang_input):
        _input_train, _output_train = yinyang_dataset_array(size=N_TRAIN, seed=np.random.randint(1e6))

        input_train = np.concatenate([input_train, _input_train], axis=-1)
        # output_train += _output_train
        output_train = np.concatenate([output_train, _output_train], axis=-1)

        _input_test, _output_test = yinyang_dataset_array(size=N_TEST, seed=np.random.randint(1e6))

        input_test = np.concatenate([input_test, _input_test], axis=-1)
        # output_test += _output_test
        output_test = np.concatenate([output_test, _output_test], axis=-1)

    data = dict(zip(["input_train", "output_train", "input_test", "output_test"],
                [input_train, output_train, input_test, output_test]))

    _tmp_method = dict(params_base) | method_param | {"n_in": 4 * n_yinyang_input,
                                                      "n_hidden": [N_HIDDEN_BASE * n_yinyang_input],
                                                      "n_out": 3 * n_yinyang_input}

    epoch_ax, _, loss, run_time = train_and_test_network(_tmp_method, model, data)

    df_learn = pd.concat([df_learn,
                          pd.DataFrame({
                            "Epoch": np.repeat(epoch_ax, params_base["n_runs"]),
                            "Sim ID": np.tile(np.arange(params_base["n_runs"]),
                                              params_base["n_test_run"]),
                            #"Accuracy": acc.flatten(),
                            "Loss": loss.flatten(),
                            "Model": model_name,
                            "Method": method_name,
                            "Number Input Datasets": n_yinyang_input
                          })], ignore_index=True)

    df_runtime = pd.concat(
       [df_runtime,
        pd.DataFrame(
            {"Runtime": run_time,
             "Sim ID": np.arange(params_base["n_runs"]),
             "Model": model_name,
             "Method": method_name,
             "Number Input Datasets": n_yinyang_input}
        )], ignore_index=True)

df_learn.to_csv(os.path.join(BASE_FOLD, "df_learn.csv"), index=False)
df_runtime.to_csv(os.path.join(BASE_FOLD, "df_runtime.csv"), index=False)
