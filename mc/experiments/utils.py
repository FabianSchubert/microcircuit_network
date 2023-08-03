import numpy as np

from mc.network import Network

from models.cs_sources import step_source_model as stp_src

from tqdm import tqdm

import time

from misc.utils import calc_loss_interp, calc_class_acc_interp

def split_lst(a, n):
    n_split = max(1,int(len(a)/n))
    res = []
    for k in range(n-1):
        res.append(a[k * n_split:(k+1) * n_split])
    res.append(a[(n-1) * n_split:])
    return res

def train_and_test_network(params, network_model, data, show_progress=False):

    input_train = data["input_train"]
    input_train_flat = input_train.flatten()
    output_train = data["output_train"]
    output_train_flat = output_train.flatten()

    input_test = data["input_test"]
    input_test_flat = input_test.flatten()
    output_test = data["output_test"]
    output_test_flat = output_test.flatten()
    
    NETWORK_NAME = params.get("name", "network")
    
    CUDA_VISIBLE_DEVICES = params.get("cuda_visible_devices", False)    

    N_IN = params["n_in"]
    N_HIDDEN = params["n_hidden"]
    N_OUT = params["n_out"]
    DT = params["dt"]

    N_RUNS = params["n_runs"]

    N_EPOCHS = params["n_epochs"]

    N_TRAIN = input_train.shape[0]
    N_TEST = input_test.shape[0]

    N_BATCH = params["n_batch"]

    N_UPDATE_PATTERN_TRAIN = int(N_EPOCHS * N_TRAIN / N_BATCH)
    N_UPDATE_PATTERN_TEST = int(N_TEST / N_BATCH)

    T_SHOW_PATTERN = params["t_show_pattern"]
    T_RUN_TRAIN = N_UPDATE_PATTERN_TRAIN * T_SHOW_PATTERN
    T_RUN_TEST = N_UPDATE_PATTERN_TEST * T_SHOW_PATTERN

    RNG_SEED = params.get("seed_ext_data", np.random.randint(1e6))

    rng = np.random.default_rng(RNG_SEED)

    SAMPLE_IDS_TRAIN = np.concatenate([rng.permutation(N_TRAIN) for _ in range(N_EPOCHS)])
    N_SAMPLE_IDS_TRAIN = SAMPLE_IDS_TRAIN.shape[0]
    UPDATE_TIMES_TRAIN = np.arange(N_UPDATE_PATTERN_TRAIN) * T_SHOW_PATTERN
    N_UPDATE_TIMES_TRAIN = UPDATE_TIMES_TRAIN.shape[0]

    SAMPLE_IDS_TEST = rng.permutation(N_TEST)
    N_SAMPLE_IDS_TEST = SAMPLE_IDS_TEST.shape[0]
    UPDATE_TIMES_TEST = np.arange(N_UPDATE_PATTERN_TEST) * T_SHOW_PATTERN
    N_UPDATE_TIMES_TEST = UPDATE_TIMES_TEST.shape[0]

    N_TEST_RUN = params["n_test_run"]

    TIMES_TEST_RUN = np.linspace(0., T_RUN_TRAIN, N_TEST_RUN+1)[:-1]

    TIMES_RECORD_TEST = np.linspace(0., T_RUN_TEST, 30 * N_UPDATE_TIMES_TEST)

    NT_SKIP_BATCH_PLAST = int(T_SHOW_PATTERN / DT)

    OPTIMIZER_PARAMS = params.get("optimizer_params", {})

    FORCE_SELF_PRED_STATE = params["force_self_pred_state"]
    FORCE_FB_ALIGN = params["force_fb_align"]

    PARAMS_TEST_RUN = [
        {
            "T": T_RUN_TEST,
            "ext_data_pop_vars": [
                (np.zeros((1, N_BATCH, N_OUT)),
                    np.array([0.]),
                    "neur_output_output_pop", "ga")
            ],
            "readout_neur_pop_vars": [
                ("neur_output_output_pop", "r", TIMES_RECORD_TEST)
            ]
        }
    ] * N_TEST_RUN

    TRAIN_READOUT = params.get("train_readout", [])

    TRAIN_READOUT_SYN = params.get("train_readout_syn", [])

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
            "data": input_test_flat,
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
            "data": output_test_flat,
            "input_id_list": SAMPLE_IDS_TEST,
            "input_times_list": UPDATE_TIMES_TEST
        }
    }

    acc = np.ndarray((N_TEST_RUN, N_RUNS))
    loss = np.ndarray((N_TEST_RUN, N_RUNS))
    run_time = np.ndarray((N_RUNS))

    net = Network(NETWORK_NAME, network_model,
                    N_IN, N_HIDDEN, N_OUT,  # network size
                    0,  # maximum number of input arrays for external input
                    0,  # spike buffer size
                    0,  # validation spike buffer size
                    [],  # spike buffer populations
                    [],  # validation spike buffer populations
                    n_batches=N_BATCH,
                    n_batches_val=N_BATCH,
                    cs_in_init=cs_in_train,
                    cs_out_init=cs_out_train,
                    cs_in_init_static_twin=cs_in_test,
                    cs_out_init_static_twin=cs_out_test,
                    plastic=True,
                    dt=DT,
                    optimizer_params=OPTIMIZER_PARAMS,
		    cuda_visible_devices=CUDA_VISIBLE_DEVICES
                    )

    readout_arrays = {}

    for run_id in tqdm(range(N_RUNS)):

        net.reinitialize()

        net.init_self_pred_state()

        t0 = time.time()

        _readout_neur_arrays, _readout_syn_arrays, readout_spikes, results_validation = net.run_sim(
                    T_RUN_TRAIN, None, TRAIN_READOUT, TRAIN_READOUT_SYN,
                    t_sign_validation=TIMES_TEST_RUN,
                    data_validation=PARAMS_TEST_RUN,
                    NT_skip_batch_plast=NT_SKIP_BATCH_PLAST,
                    force_self_pred_state=FORCE_SELF_PRED_STATE,
                    force_fb_align=FORCE_FB_ALIGN,show_progress=show_progress)

        t1 = time.time()

        out_r_test = [results_validation[k]["neur_var_rec"]["neur_output_output_pop_r"] for k in range(N_TEST_RUN)]

        SAMPLE_IDS_BATCHES_TEST = np.reshape(SAMPLE_IDS_TEST[:N_UPDATE_PATTERN_TEST*N_BATCH],(N_UPDATE_PATTERN_TEST,N_BATCH))

        _acc = []

        for k in range(N_TEST_RUN):
            _acc.append(calc_class_acc_interp(UPDATE_TIMES_TEST, TIMES_RECORD_TEST,
                            output_test[SAMPLE_IDS_BATCHES_TEST], out_r_test[k]))
        _loss = []

        for k in range(N_TEST_RUN):
            _loss.append(calc_loss_interp(UPDATE_TIMES_TEST, TIMES_RECORD_TEST,
                            output_test[SAMPLE_IDS_BATCHES_TEST], out_r_test[k]))

        _acc = np.array(_acc)
        _loss = np.array(_loss)

        acc[:,run_id] = _acc
        loss[:,run_id] = _loss
        run_time[run_id] = t1 - t0

        for key, data in (_readout_neur_arrays | _readout_syn_arrays).items():
            _d = readout_arrays.get(key, np.ndarray(data.shape + (0,)))
            readout_arrays[key] = np.concatenate([_d, np.expand_dims(data, axis=-1)], axis=-1)

    epoch_ax = np.linspace(0.,N_EPOCHS, N_TEST_RUN)

    readout_neur_arrays = {k:v for k,v in readout_arrays.items() if k in _readout_neur_arrays.keys()}
    readout_syn_arrays = {k:v for k,v in readout_arrays.items() if k in _readout_syn_arrays.keys()}

    return epoch_ax, acc, loss, readout_neur_arrays, readout_syn_arrays, run_time
