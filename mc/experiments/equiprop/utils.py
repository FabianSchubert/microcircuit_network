from network_architectures.equiprop.network import EquipropNetwork  as Network

from genn_models.cs_sources import step_source_model as stp_src

import numpy as np

import time

from tqdm import tqdm

from ..utils import calc_class_acc_interp, calc_loss_interp


def train_and_test_network(params, network_model, data, show_progress=False):
    """
        Wrapper for running tests on the equiprop model.
    """
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
    # Factor 2 because of the two phases during training.
    T_RUN_TRAIN = 2. * N_UPDATE_PATTERN_TRAIN * T_SHOW_PATTERN
    T_RUN_TEST = N_UPDATE_PATTERN_TEST * T_SHOW_PATTERN

    RNG_SEED = params.get("seed_ext_data", np.random.randint(1e6))

    rng = np.random.default_rng(RNG_SEED)

    SAMPLE_IDS_TRAIN = np.concatenate(
        [rng.permutation(N_TRAIN) for _ in range(N_EPOCHS)])
    N_SAMPLE_IDS_TRAIN = SAMPLE_IDS_TRAIN.shape[0]
    # factor 2 as above
    UPDATE_TIMES_TRAIN = 2. * np.arange(N_UPDATE_PATTERN_TRAIN) * T_SHOW_PATTERN
    N_UPDATE_TIMES_TRAIN = UPDATE_TIMES_TRAIN.shape[0]

    SAMPLE_IDS_TEST = rng.permutation(N_TEST)
    N_SAMPLE_IDS_TEST = SAMPLE_IDS_TEST.shape[0]
    UPDATE_TIMES_TEST = np.arange(N_UPDATE_PATTERN_TEST) * T_SHOW_PATTERN
    N_UPDATE_TIMES_TEST = UPDATE_TIMES_TEST.shape[0]

    N_TEST_RUN = params["n_test_run"]

    TIMES_TEST_RUN = np.linspace(0., T_RUN_TRAIN, N_TEST_RUN+1)[:-1]

    TIMES_RECORD_TEST = np.linspace(0., T_RUN_TEST, 30 * N_UPDATE_TIMES_TEST)

    # only apply changes after both phases -> factor 2.
    NT_SKIP_BATCH_PLAST = int(2. * T_SHOW_PATTERN / DT)

    OPTIMIZER_PARAMS = params.get("optimizer_params", {})

    PARAMS_TEST_RUN = [
        {
            "T": T_RUN_TEST,
            "ext_data_pop_vars": [
                (np.zeros((1, N_BATCH, N_OUT)),
                    np.array([0.]),
                    "neur_output_output_pop", "targ_mode")
            ],
            "readout_neur_pop_vars": [
                ("neur_output_output_pop", "r", TIMES_RECORD_TEST)
            ]
        }
    ] * N_TEST_RUN

    # create the data that updates the target mode for the training phases

    _update_targ_mode_phases_data = np.zeros((2*N_UPDATE_TIMES_TRAIN, N_BATCH, N_OUT))
    # start with free phase (targ_mode=0), and alternate.
    _update_targ_mode_phases_data[1::2] = 1.0
    _update_targ_mode_phases_times = np.arange(2*N_UPDATE_PATTERN_TRAIN) * T_SHOW_PATTERN

    EXT_DATA_POP_VARS_TRAIN = [
        (_update_targ_mode_phases_data,
         _update_targ_mode_phases_times,
         "neur_output_output_pop", "targ_mode")]

    for k in range(len(N_HIDDEN)):
        _update_targ_mode_phases_data = np.zeros((2*N_UPDATE_TIMES_TRAIN, N_BATCH, N_HIDDEN[k]))
        # start with free phase (targ_mode=0), and alternate.
        _update_targ_mode_phases_data[1::2] = 1.0

        EXT_DATA_POP_VARS_TRAIN.append(
            (_update_targ_mode_phases_data,
              _update_targ_mode_phases_times,
              f"neur_hidden{k}_hidden_pop", "targ_mode"))

    #### optional variable readout during training.

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
        "vars": {"idx": 0, "t_next": 0.0},
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
        "vars": {"idx": 0, "t_next": 0.0},
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
        "vars": {"idx": 0, "t_next": 0.0},
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
            "input_id_list_size": 1,
            "input_times_list_size": 1
        },
        "vars": {"idx": 0, "t_next": 0.0},
        "extra_global_params": {
            # In principle, it should not matter what you put here,
            # because the conductance ga is set to zero for the testing,
            # but just "to be sure", it is a single array of zeros,
            # rather than the actual output targets.
            "data": np.zeros(N_OUT),
            "input_id_list": np.array([0]).astype("int"),
            "input_times_list": np.array([0.])
        }
    }

    acc = np.ndarray((N_TEST_RUN, N_RUNS))
    loss = np.ndarray((N_TEST_RUN, N_RUNS))
    run_time = np.ndarray((N_RUNS))

    '''
    size_input: int,
              size_hidden: typing.List[int],
              size_output: int,
              model_def: types.ModuleType,
              optimizer_params={},
              cs_in_init: InitVar[typing.Any] = None,
              cs_out_init: InitVar[typing.Any] = None,
              cs_in_init_static_twin: InitVar[typing.Any] = None,
              cs_out_init_static_twin: InitVar[typing.Any] = None
    '''

    net = Network(NETWORK_NAME, DT,
                  N_BATCH, N_BATCH,
                  0, 0,
                  [], [],
                  N_IN, N_HIDDEN, N_OUT,
                  network_model, OPTIMIZER_PARAMS,
                  cs_in_train, cs_out_train,
                  cs_in_test, cs_out_test,
                  cuda_visible_devices=CUDA_VISIBLE_DEVICES)

    readout_arrays = {}

    for run_id in tqdm(range(N_RUNS)):

        net.reinitialize()

        net.align_fb_weights()

        t0 = time.time()

        (_readout_neur_arrays, _readout_syn_arrays,
         readout_spikes, results_validation) = net.run_sim(
                    T_RUN_TRAIN, EXT_DATA_POP_VARS_TRAIN,
                    TRAIN_READOUT, TRAIN_READOUT_SYN,
                    t_sign_validation=TIMES_TEST_RUN,
                    data_validation=PARAMS_TEST_RUN,
                    NT_skip_batch_plast=NT_SKIP_BATCH_PLAST,
                    show_progress=show_progress)

        t1 = time.time()

        out_r_test = [results_validation[k]["neur_var_rec"]["neur_output_output_pop_r"] for k in range(N_TEST_RUN)]

        SAMPLE_IDS_BATCHES_TEST = np.reshape(SAMPLE_IDS_TEST[:N_UPDATE_PATTERN_TEST*N_BATCH],(N_UPDATE_PATTERN_TEST,N_BATCH))

        _acc = []

        for k in range(N_TEST_RUN):
            _acc.append(
                calc_class_acc_interp(UPDATE_TIMES_TEST, TIMES_RECORD_TEST,
                                      output_test[SAMPLE_IDS_BATCHES_TEST],
                                      out_r_test[k])
                        )
        _loss = []

        for k in range(N_TEST_RUN):
            _loss.append(
                calc_loss_interp(UPDATE_TIMES_TEST, TIMES_RECORD_TEST,
                                 output_test[SAMPLE_IDS_BATCHES_TEST],
                                 out_r_test[k])
                        )

        _acc = np.array(_acc)
        _loss = np.array(_loss)

        acc[:, run_id] = _acc
        loss[:, run_id] = _loss
        run_time[run_id] = t1 - t0

        for key, data in (_readout_neur_arrays | _readout_syn_arrays).items():
            _d = readout_arrays.get(key, np.ndarray(data.shape + (0,)))
            readout_arrays[key] = np.concatenate(
                [_d, np.expand_dims(data, axis=-1)], axis=-1)

    epoch_ax = np.linspace(0.,N_EPOCHS, N_TEST_RUN)

    readout_neur_arrays = {k: v for k, v in readout_arrays.items() if k in _readout_neur_arrays.keys()}
    readout_syn_arrays = {k: v for k, v in readout_arrays.items() if k in _readout_syn_arrays.keys()}

    import matplotlib.pyplot as plt
    plt.ion()
    import pdb
    pdb.set_trace()

    return (epoch_ax, acc, loss, readout_neur_arrays,
            readout_syn_arrays, run_time)
