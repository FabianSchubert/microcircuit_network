import numpy as np
import pandas as pd

import os

from models.latent_equ import change_detection_spikes as network_model

from data.mnist.dataset import (input_train, output_train,
                                input_test, output_test)

from mc.network import Network
from models.cs_sources import step_source_model as stp_src

from itertools import product

data = dict(zip(["input_train", "output_train", "input_test", "output_test"],
                [input_train, output_train, input_test, output_test]))


BASE_FOLD = os.path.dirname(__file__)

N_IN = 784
N_HIDDEN = [100]
N_OUT = 10
DT = 0.5

input_train = data["input_train"][:10]
input_train_flat = input_train.flatten()
output_train = data["output_train"][:10]
output_train_flat = output_train.flatten()

N_EPOCHS = 1

N_TRAIN = input_train.shape[0]

N_BATCH = 1

N_UPDATE_PATTERN_TRAIN = int(N_EPOCHS * N_TRAIN / N_BATCH)

T_SHOW_PATTERN = 150.
T_RUN_TRAIN = N_UPDATE_PATTERN_TRAIN * T_SHOW_PATTERN

RNG_SEED = np.random.randint(1e6)

rng = np.random.default_rng(RNG_SEED)

SAMPLE_IDS_TRAIN = np.concatenate([rng.permutation(N_TRAIN) for _ in range(N_EPOCHS)])
N_SAMPLE_IDS_TRAIN = SAMPLE_IDS_TRAIN.shape[0]
UPDATE_TIMES_TRAIN = np.arange(N_UPDATE_PATTERN_TRAIN) * T_SHOW_PATTERN
N_UPDATE_TIMES_TRAIN = UPDATE_TIMES_TRAIN.shape[0]

N_TEST_RUN = 0

TIMES_TEST_RUN = np.linspace(0., T_RUN_TRAIN, N_TEST_RUN+1)[:-1]

TIMES_RECORD_TEST = None

NT_SKIP_BATCH_PLAST = int(T_SHOW_PATTERN / DT)

FORCE_SELF_PRED_STATE = False
FORCE_FB_ALIGN = False

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
        "data": input_train_flat,
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
        "data": output_train_flat,
        "input_id_list": SAMPLE_IDS_TRAIN,
        "input_times_list": UPDATE_TIMES_TRAIN
    }
}

net = Network("mnist_sample_network", network_model,
                N_IN, N_HIDDEN, N_OUT,  # network size
                0,  # maximum number of input arrays for external input
                10 * int(T_SHOW_PATTERN / DT),  # spike buffer size
                0,  # validation spike buffer size
                ["neur_hidden0_pyr_pop"],  # spike buffer populations
                [],  # validation spike buffer populations
                n_batches=N_BATCH,
                n_batches_val=N_BATCH,
                cs_in_init=cs_in_train,
                cs_out_init=cs_out_train,
                cs_in_init_static_twin=None,
                cs_out_init_static_twin=None,
                plastic=True,
                dt=DT,
                optimizer_params={},
                cuda_visible_devices=False
                )

net.init_self_pred_state()

#_readout_neur_arrays, _readout_syn_arrays, readout_spikes, results_validation
_, _, spike_results = net.run_sim(
                    T_RUN_TRAIN, None, None, None,
                    t_sign_validation=TIMES_TEST_RUN,
                    data_validation={},
                    NT_skip_batch_plast=NT_SKIP_BATCH_PLAST,
                    force_self_pred_state=FORCE_SELF_PRED_STATE,
                    force_fb_align=FORCE_FB_ALIGN,show_progress=True)

spikes_pyr = spike_results["neur_hidden0_pyr_pop"]

df_spikes = pd.DataFrame({"t": spikes_pyr[0], "id": spikes_pyr[1]})

df_spikes.to_csv(os.path.join(BASE_FOLD, "df_spikes.csv"), index=False)
