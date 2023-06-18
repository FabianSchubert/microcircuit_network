"""
This module defines the network class used to
construct an instance of the dendritic microcircuit model.
"""

import types
from dataclasses import dataclass, field, InitVar

import numpy as np
from pygenn.genn_model import GeNNModel
from tqdm import tqdm

from .layers import HiddenLayer, InputLayer, OutputLayer
from .synapses import (SynapseIPBack, SynapsePINP,
                       SynapsePPApical, SynapsePPBasal)

import typing


@dataclass
class Network:
    """
    The Network class defines the dendritic microcircuit
    model.

    Args:
        name (str):

                    Name of the network instance.

        model_def (module):

                    A module object that contains a hierarchical
                    structure of model definitions stored as
                    dictionaries. (Detailled documentation to
                    follow)

        size_input (int):

                    Number of neurons in the input layer.

        size_hidden (list):

                    List of integers defining the number.
                    of neurons in each hidden layer. Order
                    corresponds to the flow of information
                    in the "forward direction" from the
                    input to the output layer.

        size_output (int):

                    Number of neurons in the output layer.

        dt (float):

                    Simulation time step size.

        plastic (bool):

                    Flag indicating whether the network should
                    is True (default), a non-plastic twin of
                    include synaptic plasticity rules. If this
                    the network will be created that is used
                    for validation runs using
                    func: Network.run_validation.

        t_inp_max (int):

                    Maximum number of time signatures used for
                    updating the input layer and the target for
                    the output layer. This must be specified
                    because the external data is pushed to the
                    device as a whole before the start of the
                    simulation.

        spike_buffer_size (int):

                    Number of time steps in the spike recording buffer(s).
                    Note that spikes can only be read from the
                    buffer if the network was run for a number of
                    time steps at least as large as the spike buffer
                    size. If the simulation exceeds the spike buffer size,
                    spikes will be overwritten in the buffer, starting
                    from the beginning.

        spike_buffer_size_val (int):

                    Same as spike_buffer_size, but for the static network.

        spike_rec_pops (list):

                    List of name strings of neuron populations from
                    which spikes are to be recorded. Unfortunately,
                    this has to be set at the time of the initialization
                    of the network instance and can not be altered
                    for different simulation runs.

        spike_rec_pops_val (list):

                    Same as spike_rec_pops, but for the static
                    twin network used vor validation.

        optimizer_params (dict):

                    A dictionary with names of neuron or synapse populations
                    as keys, each holding a dictionary with parameters for the
                    optimizer to be used for the biases (neuron populations) or
                    the weights (synapse population). Each of these dicts should
                    be of the form:
                    {"optimizer": one of {"sgd", "sgd_momentum", "adam" (default)},
                     "params": parameters for the chosen optimizer as a dict}.
                    If you do not speficy these parameters for a population, the
                    network will use th default (adam).
    """

    name: str
    model_def: types.ModuleType
    size_input: int
    size_hidden: list
    size_output: int
    t_inp_max: int
    spike_buffer_size: int
    spike_buffer_size_val: int
    spike_rec_pops: list = field(default_factory=list)
    spike_rec_pops_val: list = field(default_factory=list)
    dt: float = 0.1
    plastic: bool = True
    t_inp_static_max: int = 0
    n_batches: int = 2
    n_batches_val: int = 1
    cs_in_init: InitVar[typing.Any] = None
    cs_out_init: InitVar[typing.Any] = None
    cs_in: typing.Any = field(init=False)
    cs_out: typing.Any = field(init=False)
    cs_in_init_static_twin: InitVar[typing.Any] = None
    cs_out_init_static_twin: InitVar[typing.Any] = None
    optimizer_params: dict = field(default_factory=lambda: {})

    def __post_init__(self, cs_in_init, cs_out_init,
                      cs_in_init_static_twin, cs_out_init_static_twin):

        self.genn_model = GeNNModel("float", self.name, backend="CUDA")

        self.genn_model.batch_size = self.n_batches

        self.genn_model.dT = self.dt

        self.n_hidden_layers = len(self.size_hidden)
        # there must be at least one hidden layer.
        assert self.n_hidden_layers > 0

        self.layers = []

        self.layers.append(
            InputLayer("input",
                       self.genn_model,
                       self.model_def.neurons.input.mod_dat,
                       self.size_input,
                       optimizer_params=self.optimizer_params))

        for k in range(self.n_hidden_layers - 1):
            _l_size = self.size_hidden[k]
            _l_next_size = self.size_hidden[k + 1]

            self.layers.append(
                HiddenLayer(f'hidden{k}',
                            self.genn_model,
                            self.model_def.neurons.pyr.mod_dat,
                            self.model_def.neurons.int.mod_dat,
                            self.model_def.synapses.IP.mod_dat,
                            self.model_def.synapses.PI.mod_dat,
                            _l_size, _l_next_size,
                            plastic=self.plastic,
                            read_only_weights=self.plastic,
                            optimizer_params=self.optimizer_params))


        self.layers.append(
            HiddenLayer(f'hidden{self.n_hidden_layers - 1}',
                        self.genn_model,
                        self.model_def.neurons.pyr.mod_dat,
                        self.model_def.neurons.int.mod_dat,
                        self.model_def.synapses.IP.mod_dat,
                        self.model_def.synapses.PI.mod_dat,
                        self.size_hidden[self.n_hidden_layers - 1],
                        self.size_output,
                        plastic=self.plastic,
                        read_only_weights=self.plastic,
                        optimizer_params=self.optimizer_params))

        self.layers.append(
            OutputLayer("output",
                        self.genn_model,
                        self.model_def.neurons.output.mod_dat,
                        self.size_output,
                        optimizer_params=self.optimizer_params))



        self.update_neur_pops()


        ##############################################
        # cross-layer synapse populations

        self.cross_layer_syn_pops = []

        _n_in = self.neur_pops["neur_input_input_pop"].size
        self.cross_layer_syn_pops.append(
            SynapsePINP(_n_in,
                        **self.model_def.synapses.PINP.mod_dat
                        ).connect_pops(
                'syn_input_input_pop_to_hidden0_pyr_pop',
                self.genn_model,
                self.neur_pops["neur_hidden0_pyr_pop"],
                self.neur_pops["neur_input_input_pop"],
                plastic=self.plastic,
                read_only=self.plastic,
                optimizer_params=self.optimizer_params
            )
        )

        for k in range(self.n_hidden_layers - 1):
            _l = self.layers[k + 1]
            _l_next = self.layers[k + 2]

            _n_in = _l.neur_pops["pyr_pop"].size

            self.cross_layer_syn_pops.append(
                SynapsePPBasal(_n_in,
                               **self.model_def.synapses.PPBasal.mod_dat
                               ).connect_pops(
                    f'syn_{_l.name}_pyr_pop_to_{_l_next.name}_pyr_pop',
                    self.genn_model,
                    _l_next.neur_pops["pyr_pop"],
                    _l.neur_pops["pyr_pop"],
                    plastic=self.plastic,
                    read_only=self.plastic,
                    optimizer_params=self.optimizer_params
                )
            )

            _n_in = _l_next.neur_pops["pyr_pop"].size

            self.cross_layer_syn_pops.append(
                SynapsePPApical(_n_in,
                                **self.model_def.synapses.PPApical.mod_dat
                                ).connect_pops(
                    f'syn_{_l_next.name}_pyr_pop_to_{_l.name}_pyr_pop',
                    self.genn_model,
                    _l.neur_pops["pyr_pop"],
                    _l_next.neur_pops["pyr_pop"],
                    plastic=self.plastic,
                    read_only=self.plastic,
                    optimizer_params=self.optimizer_params
                )
            )

            self.cross_layer_syn_pops.append(
                SynapseIPBack(**self.model_def.synapses.IPBack.mod_dat).connect_pops(
                    f'syn_{_l_next.name}_pyr_pop_to_{_l.name}_int_pop',
                    self.genn_model,
                    _l.neur_pops["int_pop"],
                    _l_next.neur_pops["pyr_pop"],
                    plastic=self.plastic,
                    read_only=self.plastic,
                    optimizer_params=self.optimizer_params
                )
            )

        _n_in = self.layers[-2].neur_pops["pyr_pop"].size

        self.cross_layer_syn_pops.append(
            SynapsePPBasal(_n_in,
                           **self.model_def.synapses.PPBasal.mod_dat
                           ).connect_pops(
                f'syn_{self.layers[-2].name}_pyr_pop_to_output_output_pop',
                self.genn_model,
                self.layers[-1].neur_pops["output_pop"],
                self.layers[-2].neur_pops["pyr_pop"],
                plastic=self.plastic,
                read_only=self.plastic,
                optimizer_params=self.optimizer_params
            )
        )

        _n_in = self.layers[-1].neur_pops["output_pop"].size

        self.cross_layer_syn_pops.append(
            SynapsePPApical(_n_in,
                            **self.model_def.synapses.PPApical.mod_dat
                            ).connect_pops(
                f'syn_output_output_pop_to_{self.layers[-2].name}_pyr_pop',
                self.genn_model,
                self.layers[-2].neur_pops["pyr_pop"],
                self.layers[-1].neur_pops["output_pop"],
                plastic=self.plastic,
                read_only=self.plastic,
                optimizer_params=self.optimizer_params
            )
        )

        self.cross_layer_syn_pops.append(
            SynapseIPBack( **self.model_def.synapses.IPBack.mod_dat).connect_pops(
                f'syn_output_pyr_pop_to_{self.layers[-2].name}_int_pop',
                self.genn_model,
                self.layers[-2].neur_pops["int_pop"],
                self.layers[-1].neur_pops["output_pop"],
                plastic=self.plastic,
                read_only=self.plastic,
                optimizer_params=self.optimizer_params
            )
        )

        self.update_syn_pops()


        #
        #################################################

        ############################
        # current sources
        if cs_in_init:
            self.add_input_current_source(cs_in_init["model"],
                                          cs_in_init["params"],
                                          cs_in_init["vars"],
                                          cs_in_init["extra_global_params"])
        else:
            self.cs_in = None

        if cs_out_init:
            self.add_output_current_source(cs_out_init["model"],
                                           cs_out_init["params"],
                                           cs_out_init["vars"],
                                           cs_out_init["extra_global_params"])
        else:
            self.cs_out = None
        ############################

        ############################
        # set up the spike recording
        if self.spike_rec_pops is None:
            self.spike_rec_pops = []

        for _pop in self.spike_rec_pops:
            self.neur_pops[_pop].spike_recording_enabled = True

        ############################

        self.genn_model.build()



        # initialize the extra global parameters required for
        # putting the input to the gpu
        #_t_sign_init = np.zeros(self.t_inp_max)

        #_inp_pop = self.neur_pops["neur_input_input_pop"]

        #_u_input_init = np.zeros((_inp_pop.size * self.n_batches * self.t_inp_max))

        #_inp_pop.set_extra_global_param("u", _u_input_init)
        #_inp_pop.set_extra_global_param("t_sign", _t_sign_init)
        #_inp_pop.set_extra_global_param("size_u", self.size_input)
        #_inp_pop.set_extra_global_param("size_t_sign", self.t_inp_max)
        #_inp_pop.set_extra_global_param("batch_size", self.n_batches)

        #_output_pop = self.neur_pops["neur_output_output_pop"]

        #_u_trg_init = np.zeros((_output_pop.size * self.n_batches * self.t_inp_max))

        #_output_pop.set_extra_global_param("u_trg", _u_trg_init)
        #_output_pop.set_extra_global_param("t_sign", _t_sign_init)
        #_output_pop.set_extra_global_param("size_u_trg", self.size_output)
        #_output_pop.set_extra_global_param("size_t_sign", self.t_inp_max)
        #_output_pop.set_extra_global_param("batch_size", self.n_batches)

        self.genn_model.load(num_recording_timesteps=self.spike_buffer_size)

        self.norm_weights()

        # only if this instance is plastic, we create a static
        # twin of the network as a member variable of itself
        # (otherwise, calling create_static_twin() would lead
        # to an infinite recursive loop).
        if self.plastic:
            self.create_static_twin(cs_in_init=cs_in_init_static_twin,
                                    cs_out_init=cs_out_init_static_twin)

    def norm_weights(self):
         ############################
        # normalize weights
        for synpop in self.syn_pops.values():
            norm = synpop.norm_after_init
            assert norm in (False, "lin", "lin_inv", "sqrt", "sqrt_inv"), \
                """Error: wrong synapse normalisation argument"""
            if norm:
                weightview = synpop.vars["g"].view
                synpop.pull_var_from_device("g")

                if norm == "lin":
                    normfact = synpop.src.size
                elif norm == "lin_inv":
                    normfact = synpop.trg.size
                elif norm == "sqrt":
                    normfact = np.sqrt(synpop.src.size)
                else:
                    normfact = np.sqrt(synpop.trg.size)

                weightview[:] = weightview[:] / normfact
                synpop.push_var_to_device("g")
        ############################

    def reinitialize(self):

        self.genn_model.reinitialise()
        self.norm_weights()

        if self.plastic:
            self.static_twin_net.reinitialize()


    def update_neur_pops(self):
        """updates the dictionary self.neur_pops
        by running through all layers
        to get a full list of all neuron populations."""

        self.neur_pops = {}
        for layer in self.layers:
            for pop in layer.neur_pops.values():
                self.neur_pops[pop.name] = pop

    def update_syn_pops(self):
        """updates the dictionary self.syn_pops
        by running through all layers and cross-layer
        synapse populations to get a full list
        of all synapse populations."""

        self.syn_pops = {}

        for layer in self.layers:
            for pop in layer.syn_pops.values():
                self.syn_pops[pop.name] = pop

        for pop in self.cross_layer_syn_pops:
            self.syn_pops[pop.name] = pop

    def create_static_twin(self, cs_in_init=None, cs_out_init=None):
        """create a non-plastic copy of the
        network by removing the plastic component
        in the weight update rules for the copy.
        The copy self.static_twin will be an
        instance of the network itself."""

        self.static_twin_net = Network(
            f"static_twin_{self.name}", self.model_def,
            self.size_input, self.size_hidden, self.size_output,
            self.t_inp_static_max,
            self.spike_buffer_size_val, 0,
            spike_rec_pops=self.spike_rec_pops_val,
            dt=self.dt, plastic=False, n_batches=self.n_batches_val,
            cs_in_init=cs_in_init,
            cs_out_init=cs_out_init
        )

    def add_input_current_source(self, cs_model,
                                 cs_params, cs_vars,
                                 cs_extra_global_params):

        self.cs_in = self.genn_model.add_current_source("cs_in", cs_model,
                                                        self.neur_pops["neur_input_input_pop"],
                                                        cs_params, cs_vars)
        for pkey, pval in cs_extra_global_params.items():
            self.cs_in.set_extra_global_param(pkey, pval)

    def add_output_current_source(self, cs_model,
                                  cs_params, cs_vars,
                                  cs_extra_global_params):

        self.cs_out = self.genn_model.add_current_source("cs_out", cs_model,
                                                         self.neur_pops["neur_output_output_pop"],
                                                         cs_params, cs_vars)
        for pkey, pval in cs_extra_global_params.items():
            self.cs_out.set_extra_global_param(pkey, pval)

    def run_sim(self, T,
                #t_sign,
                #ext_data_input, ext_data_output,
                ext_data_pop_vars, readout_neur_pop_vars,
                readout_syn_pop_vars,
                t_sign_validation=None,
                data_validation=None,
                show_progress=True,
                show_progress_val=True,
                NT_skip_batch_plast=1,
                force_self_pred_state=False,
                force_fb_align=False):
        """
        run_sim simulates the network and allows the user
        to provide input data as well as specify targets
        for readout/recording.

        Args:

            t_sign (numpy.ndarray):

                        Time signatures for the update
                        of the input rates and the values
                        of u_trg in the output population.

            ext_data_pop_vars (list):

                        Additional data and time signatures targeting
                        neuron population variables in the network.

                        ext_data_pop_vars should be a list tuples of the form
                        (numpy_data, numpy_times, target_pop, target_var).

                        numpy_times is a list of time indices whose length
                        should be the same as the first dimension of
                        numpy_data.

            readout_neur_pop_vars (list):

                        List of tuples of the form
                        (readout_pop, readout_var, numpy_times).
                        readout_pop is the name of the population
                        to be read from & readout var specifies the
                        variable to be pulled.
                        numpy_times is an array specifying
                        the readout times.

            readout_syn_pop_vars (list):

                        Same structure as readout_neur_pop_vars.

            t_sign_validation (np.ndarray):

                        Time signatures for calling a validation run.

            data_validation (list):

                        List of dictionaries, each of which holds
                        arguments for a validation run executed at
                        one of the time steps given in t_sign_validation.
                        The arguments required are a subset of the arguments
                        provided to run_sim, namely
                        {T, t_sign, ext_data_input, ext_data_pop_vars,
                        readout_neur_pop_vars}.
                        Note that this does not include any "target data",
                        e.g. for the readout. The validation runs only yield
                        the simulation data specified in readout_neur_pop_vars,
                        and it is the user's responsibility to construct
                        meaningful validation measures from the data that is
                        returned. Furthermore, there is no option for
                        recording synapses, since they are kept fixed
                        during the validation.

            show_progress (bool):

                        If true, a tqdm progress bar is displayed during
                        the simulation run.

            show_progress_val (bool):

                        Same as show_progress, but for the static twin.

            NT_skip_batch_plast (int):

                        Number of time steps between an update where
                        the synaptic changes stored in the individual batch
                        instances under dg are reduced (averaged over the batches)
                        and added to the weights. After this, the synaptic
                        dg variables are set back to zero. The default
                        value of 1 means that this happens every time
                        step (generally not recommended for performance).
        """

        NT = int(T / self.dt)

        input_views = []

        # List of indices storing the current "head" index from
        # which to load when the next time data is pushed to a population.
        idx_data_heads = []

        #t_sign = np.ndarray((0)) if t_sign is None else t_sign

        ext_data_pop_vars = [] if ext_data_pop_vars is None else ext_data_pop_vars

        n_inputs = len(ext_data_pop_vars)

        readout_neur_pop_vars = [] if readout_neur_pop_vars is None else readout_neur_pop_vars

        readout_syn_pop_vars = [] if readout_syn_pop_vars is None else readout_syn_pop_vars

        #ext_data_input = np.ndarray((0)) if ext_data_input is None else ext_data_input
        #ext_data_output = np.ndarray((0)) if ext_data_output is None else ext_data_output

        # List for copies of time signatures for extra input data.
        time_signatures_ext_data = []

        if data_validation:

            assert t_sign_validation is not None, \
                """Error: validation data was provided,
                but no time signatures were given.
                """

            assert t_sign_validation.shape[0] == len(data_validation), \
                """Error: Number of validation time signatures
                does not match number of validation datasets."""

            # List to store the results of the validation runs.
            results_validation = []

            # Counter for keeping track of the next validation run
            idx_validation_runs = 0

        for ext_dat, numpy_times, target_pop, target_var in ext_data_pop_vars:
            assert ext_dat.ndim == (2 if self.n_batches == 1 else 3), \
                """Input data array does
                not have appropriate dimensions.
                Needs to be 2 if n_batches = 1, else 3"""

            _target_pop = self.neur_pops[target_pop]
            assert ext_dat.shape[-1] == _target_pop.size, \
                """Input data size in last dimension and
                input population size do not match."""

            if self.n_batches > 1:
                assert ext_dat.shape[1] == self.n_batches, \
                """
                Input data in second dimension and
                batch size do not match.
                """

            assert len(
                numpy_times) > 0, """Error: Passed empty input
                data time span list."""

            assert np.all(numpy_times[:-1] < numpy_times[1:]), \
                """Error: Input times are not correctly
                ordered"""

            assert numpy_times.shape[0] == ext_dat.shape[0], \
                """Error: Size of input data does not match
                number of time signatures."""

            time_signatures_ext_data.append(np.array(numpy_times))

            idx_data_heads.append(0)

            input_views.append(_target_pop.vars[target_var].view)

        readout_views = {}

        readout_neur_arrays = {}

        readout_syn_arrays = {}

        time_signatures_readout_neur_pop = []
        idx_readout_neur_pop_heads = []

        time_signatures_readout_syn_pop = []
        idx_readout_syn_pop_heads = []

        for readout_pop, readout_var, t_sign_readout in readout_neur_pop_vars:
            _dict_name = f'{readout_pop}_{readout_var}'
            _view = self.neur_pops[readout_pop].vars[readout_var].view
            readout_views[_dict_name] = _view
            readout_neur_arrays[_dict_name] = np.ndarray(
                (t_sign_readout.shape[0], self.n_batches, self.neur_pops[readout_pop].size))

            time_signatures_readout_neur_pop.append(np.array(t_sign_readout))
            idx_readout_neur_pop_heads.append(0)

        for readout_pop, readout_var, t_sign_readout in readout_syn_pop_vars:
            _dict_name = f'{readout_pop}_{readout_var}'
            _trg_size = self.syn_pops[readout_pop].trg.size
            _src_size = self.syn_pops[readout_pop].src.size
            readout_syn_arrays[_dict_name] = np.ndarray((t_sign_readout.shape[0],
                                                         _trg_size,
                                                         _src_size))

            time_signatures_readout_syn_pop.append(np.array(t_sign_readout))
            idx_readout_syn_pop_heads.append(0)

        ####################################################
        # prepare some internal state variables / parameters

        # reset some internal state variables
        self.genn_model.t = 0.0
        self.genn_model.timestep = 0

        # reset inSyn to zero, so that changes in the input
        # are correctly tracked.
        for _pop in self.syn_pops.values():
            _pop.in_syn = np.zeros(_pop.trg.size)
            _pop.push_in_syn_to_device()

        #_inp_pop = self.neur_pops["neur_input_input_pop"]

        #_inp_pop.extra_global_params["u"].view[:
                                              # ext_data_input.shape[0]] = ext_data_input
        #_inp_pop.extra_global_params["t_sign"].view[:t_sign.shape[0]] = t_sign
        #_inp_pop.extra_global_params["size_t_sign"].view[:] = t_sign.shape[0]

        #_inp_pop.push_extra_global_param_to_device(
        #    "u", self.size_input * self.n_batches * self.t_inp_max)
        #_inp_pop.push_extra_global_param_to_device("t_sign", self.t_inp_max)

        # reset the input index counter before running the simulation
        #_inp_pop.vars["idx_dat"].view[:] = 0
        #_inp_pop.push_var_to_device("idx_dat")

        if self.plastic:
            pass
            #_output_pop = self.neur_pops["neur_output_output_pop"]

            #_output_pop.extra_global_params["u_trg"].view[:
            #                                              ext_data_output.shape[0]] = ext_data_output
            #_output_pop.extra_global_params["t_sign"].view[:t_sign.shape[0]] = t_sign
            #_output_pop.extra_global_params["size_t_sign"].view[:] = t_sign.shape[0]
            #
            #_output_pop.push_extra_global_param_to_device(
            #    "u_trg", self.size_output * self.n_batches * self.t_inp_max)
            #_output_pop.push_extra_global_param_to_device(
            #    "t_sign", self.t_inp_max)

            #_output_pop.vars["idx_dat"].view[:] = 0
            #_output_pop.push_var_to_device("idx_dat")

        ####################################################


        for t in tqdm(range(NT), disable=not show_progress, leave=self.plastic):

            # manual variable manipulation

            self.push_ext_data(t, n_inputs, time_signatures_ext_data, ext_data_pop_vars, input_views, idx_data_heads)

            # check if validation run should be
            # executed in this time step.
            if data_validation and self.plastic:
                if (idx_validation_runs < t_sign_validation.shape[0]
                        and t_sign_validation[idx_validation_runs] <= t*self.dt):

                    results_validation.append(
                        self.run_validation(**data_validation[idx_validation_runs],
                                            show_progress=show_progress_val))

                    idx_validation_runs += 1

            self.genn_model.step_time()

            if self.plastic and (t%NT_skip_batch_plast == 0):
                self.genn_model.custom_update("WeightChangeBatchReduce")
                self.genn_model.custom_update("BiasChangeBatchReduce")
                self.genn_model.custom_update("Plast")

                if force_fb_align:
                    self.align_fb_weights()
                if force_self_pred_state:
                    self.init_self_pred_state()

            self.pull_neur_var_data(t, readout_neur_pop_vars, time_signatures_readout_neur_pop,
                                    readout_neur_arrays, readout_views, idx_readout_neur_pop_heads)

            self.pull_syn_var_data(t, readout_syn_pop_vars, time_signatures_readout_syn_pop,
                              readout_syn_arrays, idx_readout_syn_pop_heads)

        readout_spikes = {}

        if len(self.spike_rec_pops) > 0:
            self.genn_model.pull_recording_buffers_from_device()

            for _pop in self.spike_rec_pops:
                readout_spikes[_pop] = self.neur_pops[_pop].spike_recording_data

        if data_validation:
            return readout_neur_arrays, readout_syn_arrays, readout_spikes, results_validation
        return readout_neur_arrays, readout_syn_arrays, readout_spikes
    
    def get_weights(self):

        weights = {}
        
        for key, syn_pop in self.syn_pops.items():
            view = syn_pop.vars["g"].view

            syn_pop.pull_var_from_device("g")

            # copy weights into new numpy array
            # (just storing a reference in the list could lead to
            # very unpredictable behavior)
            weights[key] = np.array(view)
        
        return weights
    
    def set_weights(self, weights):

        for key, value in weights.items():
            _syn_pop = self.syn_pops[key]

            _view = _syn_pop.vars["g"].view
            _view[:] = value

            _syn_pop.push_var_to_device("g")

    def get_biases(self):

        biases = {}
        
        for key, neur_pop in self.neur_pops.items():
            view = neur_pop.vars["b"].view

            neur_pop.pull_var_from_device("b")

            biases[key] = np.array(view)

        return biases

    def set_biases(self, biases):

        for key, value in biases.items():
            _neur_pop = self.neur_pops[key]

            _view = _neur_pop.vars["b"].view
            _view[:] = value

            _neur_pop.push_var_to_device("b")


    def run_validation(self, T, #t_sign,
                       #ext_data_input,
                       ext_data_pop_vars,
                       readout_neur_pop_vars, show_progress=True):
        '''
        This method runs a network simulation on the
        static twin after copying the current weight
        configuration to the twin. The output can
        be used e.g. to validate network performance.
        '''

        self.static_twin_net.genn_model.reinitialise()

        # store all the weights and biases in the network
        # and copy them over to the static twin

        weights = self.get_weights()
        self.static_twin_net.set_weights(weights)

        biases = self.get_biases()
        self.static_twin_net.set_biases(biases)

        result_neur_arrays, _, result_spikes = self.static_twin_net.run_sim(
            T, #t_sign,
            #ext_data_input, np.ndarray(0),
            ext_data_pop_vars, readout_neur_pop_vars, [], show_progress=show_progress)

        return {"neur_var_rec": result_neur_arrays,
                "spike_rec": result_spikes}

    def push_ext_data(self, t, n_inputs, time_signatures_ext_data, ext_data_pop_vars, input_views, idx_data_heads):

        for k in range(n_inputs):

            # if the array of time signatures is not empty...
            if time_signatures_ext_data[k].shape[0] > 0:

                # check if the current time is equal to the
                # current first element in the
                # array of time signatures.
                if time_signatures_ext_data[k][0] <= t*self.dt:
                    input_views[k][:] = ext_data_pop_vars[k][0][idx_data_heads[k]]

                    self.neur_pops[ext_data_pop_vars[k][2]].push_var_to_device(
                        ext_data_pop_vars[k][3])

                    idx_data_heads[k] += 1

                    # remove the current first element of the
                    # time signatures after use.
                    time_signatures_ext_data[k] = time_signatures_ext_data[k][1:]

    def pull_neur_var_data(self, t, readout_neur_pop_vars, time_signatures_readout_neur_pop,
                           readout_neur_arrays, readout_views, idx_readout_neur_pop_heads):

        for k, (readout_pop, readout_var, _) in enumerate(readout_neur_pop_vars):

            if time_signatures_readout_neur_pop[k].shape[0] > 0:

                if time_signatures_readout_neur_pop[k][0] <= t*self.dt:
                    self.neur_pops[readout_pop].pull_var_from_device(
                        readout_var)
                    _dict_name = f'{readout_pop}_{readout_var}'
                    readout_neur_arrays[_dict_name][idx_readout_neur_pop_heads[k]
                    ] = readout_views[_dict_name]

                    idx_readout_neur_pop_heads[k] += 1

                    time_signatures_readout_neur_pop[k] = time_signatures_readout_neur_pop[k][1:]

    def pull_syn_var_data(self, t, readout_syn_pop_vars, time_signatures_readout_syn_pop,
                          readout_syn_arrays, idx_readout_syn_pop_heads):

        for k, (readout_pop, readout_var, _) in enumerate(readout_syn_pop_vars):

            if time_signatures_readout_syn_pop[k].shape[0] > 0:

                if time_signatures_readout_syn_pop[k][0] <= t*self.dt:
                    self.syn_pops[readout_pop].pull_var_from_device(
                        readout_var)
                    _dict_name = f'{readout_pop}_{readout_var}'

                    readout_syn_arrays[_dict_name][idx_readout_syn_pop_heads[k]] = np.reshape(
                        self.syn_pops[readout_pop].get_var_values(
                            readout_var),
                        readout_syn_arrays[_dict_name].shape[1:], order='F')

                    idx_readout_syn_pop_heads[k] += 1

                    time_signatures_readout_syn_pop[k] = time_signatures_readout_syn_pop[k][1:]

    def init_self_pred_state(self):
        """
        Modify the PI and IP weights such that the
        network is in the self-predicting state,
        given the current state of the PP-forward
        and PP-backward weights.
        This can only be called after the genn model
        was loaded.
        """

        for _l in self.layers:

            if type(_l) == HiddenLayer:

                # really bad approach - skip the "hidden" part
                # the layer name, which is always "hidden<idx>"
                idx = int(_l.name[6:])

                _name_this_layer = _l.name

                if idx < (self.n_hidden_layers - 1):
                    _name_next_layer = f'hidden{idx + 1}'
                    _name_next_pyr = "pyr_pop"
                else:
                    _name_next_layer = "output"
                    _name_next_pyr = "output_pop"

                _name_pp_fwd = f'syn_{_name_this_layer}_pyr_pop_to_{_name_next_layer}_{_name_next_pyr}'
                _pp_fwd = self.syn_pops[_name_pp_fwd]
                _synview_pp_fwd = _pp_fwd.vars["g"].view

                _name_pp_back = f'syn_{_name_next_layer}_{_name_next_pyr}_to_{_name_this_layer}_pyr_pop'
                _pp_back = self.syn_pops[_name_pp_back]
                _synview_pp_back = _pp_back.vars["g"].view

                _pi = _l.syn_pops["int_pop_to_pyr_pop"]
                _synview_pi = _pi.vars["g"].view
                _ip = _l.syn_pops["pyr_pop_to_int_pop"]
                _synview_ip = _ip.vars["g"].view

                # set the PI weights to the negative of the PP_back weights
                _pp_back.pull_var_from_device("g")
                _synview_pi[:] = -_synview_pp_back
                _pi.push_var_to_device("g")

                # set the IP weights to the PP_fwd weights
                _pp_fwd.pull_var_from_device("g")
                _synview_ip[:] = _synview_pp_fwd
                _ip.push_var_to_device("g")

                # set biases of int pop in current layer to biases in next layer
                _next_pyr_pop = self.neur_pops[f'neur_{_name_next_layer}_{_name_next_pyr}']
                _int_pop = self.neur_pops[f'neur_{_name_this_layer}_int_pop']

                _next_pyr_pop.pull_var_from_device("b")
                _int_pop.vars["b"].view[:] = np.array(_next_pyr_pop.vars["b"].view)
                _int_pop.push_var_to_device("b")

    def align_fb_weights(self):

        for _l in self.layers:

            if type(_l) == HiddenLayer:

                # really bad approach - skip the "hidden" part
                # the layer name, which is always "hidden<idx>"
                idx = int(_l.name[6:])

                _name_this_layer = _l.name

                if idx < (self.n_hidden_layers - 1):
                    _name_next_layer = f'hidden{idx + 1}'
                    _name_next_pyr = "pyr_pop"
                else:
                    _name_next_layer = "output"
                    _name_next_pyr = "output_pop"

                _name_pp_fwd = f'syn_{_name_this_layer}_pyr_pop_to_{_name_next_layer}_{_name_next_pyr}'
                _pp_fwd = self.syn_pops[_name_pp_fwd]
                _synview_pp_fwd = _pp_fwd.vars["g"].view

                _w_pp_fwd = np.reshape(np.array(_synview_pp_fwd), (_pp_fwd.src.size, _pp_fwd.trg.size))

                _name_pp_back = f'syn_{_name_next_layer}_{_name_next_pyr}_to_{_name_this_layer}_pyr_pop'
                _pp_back = self.syn_pops[_name_pp_back]
                _synview_pp_back = _pp_back.vars["g"].view

                _synview_pp_back[:] = _w_pp_fwd.T.flatten()
                _pp_back.push_var_to_device("g")

