"""
This module defines the network class used to
construct an instance of the dendritic microcircuit model.
"""

import types
from dataclasses import dataclass, field

import numpy as np
from pygenn.genn_model import GeNNModel
from tqdm import tqdm

from .layers import HiddenLayer, InputLayer, OutputLayer
from .synapses import (SynapseIPBack, SynapsePINP,
                       SynapsePPApical, SynapsePPBasal)


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

    def __post_init__(self):

        self.genn_model = GeNNModel("float", self.name, backend="CUDA")

        self.genn_model.dT = self.dt

        self.n_hidden_layers = len(self.size_hidden)

        self.layers = []

        self.layers.append(
            InputLayer("input",
                       self.genn_model,
                       self.model_def.neurons.input.mod_dat,
                       self.size_input))

        # there must be at least one hidden layer.
        assert len(self.size_hidden) > 0

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
                            plastic=self.plastic))

        self.layers.append(
            HiddenLayer(f'hidden{self.n_hidden_layers - 1}',
                        self.genn_model,
                        self.model_def.neurons.pyr.mod_dat,
                        self.model_def.neurons.int.mod_dat,
                        self.model_def.synapses.IP.mod_dat,
                        self.model_def.synapses.PI.mod_dat,
                        self.size_hidden[self.n_hidden_layers - 1],
                        self.size_output,
                        plastic=self.plastic))

        self.layers.append(
            OutputLayer("output",
                        self.genn_model,
                        self.model_def.neurons.output.mod_dat,
                        self.size_output))

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
                plastic=self.plastic
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
                    plastic=self.plastic
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
                    plastic=self.plastic
                )
            )

            self.cross_layer_syn_pops.append(
                SynapseIPBack(**self.model_def.synapses.IPBack.mod_dat).connect_pops(
                    f'syn_{_l_next.name}_pyr_pop_to_{_l.name}_int_pop',
                    self.genn_model,
                    _l.neur_pops["int_pop"],
                    _l_next.neur_pops["pyr_pop"],
                    plastic=self.plastic
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
                plastic=self.plastic
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
                plastic=self.plastic
            )
        )

        self.cross_layer_syn_pops.append(
            SynapseIPBack( **self.model_def.synapses.IPBack.mod_dat).connect_pops(
                f'syn_output_pyr_pop_to_{self.layers[-2].name}_int_pop',
                self.genn_model,
                self.layers[-2].neur_pops["int_pop"],
                self.layers[-1].neur_pops["output_pop"],
                plastic=self.plastic
            )
        )

        self.update_syn_pops()


        #
        #################################################

        ############################
        # set up the spike recording

        for _pop in self.spike_rec_pops:
            self.neur_pops[_pop].spike_recording_enabled = True

        ############################

        self.genn_model.build()



        # initialize the extra global parameters required for
        # putting the input to the gpu
        _t_sign_init = np.zeros(self.t_inp_max)

        _inp_pop = self.neur_pops["neur_input_input_pop"]

        _u_input_init = np.zeros((_inp_pop.size * self.t_inp_max))

        _inp_pop.set_extra_global_param("u", _u_input_init)
        _inp_pop.set_extra_global_param("t_sign", _t_sign_init)
        _inp_pop.set_extra_global_param("size_u", self.size_input)
        _inp_pop.set_extra_global_param("size_t_sign", self.t_inp_max)

        _output_pop = self.neur_pops["neur_output_output_pop"]

        _u_trg_init = np.zeros((_output_pop.size * self.t_inp_max))

        _output_pop.set_extra_global_param("u_trg", _u_trg_init)
        _output_pop.set_extra_global_param("t_sign", _t_sign_init)
        _output_pop.set_extra_global_param("size_u_trg", self.size_output)
        _output_pop.set_extra_global_param("size_t_sign", self.t_inp_max)

        self.genn_model.load(num_recording_timesteps=self.spike_buffer_size)

        ############################
        # normalize weights
        for synpop in self.syn_pops.values():
            norm = synpop.norm_after_init
            assert norm in (False, "lin", "sqrt") , \
                """Error: wrong synapse normalisation argument"""
            if norm:
                weightview = synpop.vars["g"].view
                synpop.pull_var_from_device("g")
                
                if norm == "lin":
                    normfact = synpop.src.size
                else:
                    normfact = np.sqrt(synpop.src.size)

                weightview[:] = weightview[:] / normfact
                synpop.push_var_to_device("g")
        ############################

        # only if this instance is plastic, we create a static
        # twin of the network as a member variable of itself
        # (otherwise, calling create_static_twin() would lead
        # to an infinite recursive loop).
        if self.plastic:
            self.create_static_twin()

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

    def create_static_twin(self):
        """create a non-plastic copy of the
        network by removing the plastic component
        in the weight update rules for the copy.
        The copy self.static_twin will be an
        instance of the network itself."""

        self.static_twin_net = Network(
            "static_twin", self.model_def,
            self.size_input, self.size_hidden, self.size_output,
            self.t_inp_static_max,
            self.spike_buffer_size_val, 0,
            spike_rec_pops=self.spike_rec_pops_val,
            dt=self.dt, plastic=False)

    def run_sim(self, T,
                t_sign,
                ext_data_input, ext_data_output,
                ext_data_pop_vars, readout_neur_pop_vars,
                readout_syn_pop_vars,
                t_sign_validation=None,
                data_validation=None,
                show_progress=True,
                show_progress_val=True):
        """
        run_sim simulates the network and allows the user
        to provide input data as well as specify targets
        for readout/recording.

        Args:

            t_sign (numpy.ndarray):

                        Time signatures for the update
                        of the input rates and the values
                        of u_trg in the output population.

            ext_data_input (numpy.ndarray):

                        Data array for the input rates.
                        Should be of size (len(t_sign) * size_input).

            ext_data_output (numpy.ndarray):

                        Data array for the u_trg variables.
                        Should be of size (len(t_sign) * size_output).

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
        """

        input_views = []

        # List of indices storing the current "head" index from
        # which to load when the next time data is pushed to a population.
        idx_data_heads = []

        t_sign = np.ndarray((0)) if t_sign is None else t_sign

        ext_data_pop_vars = [] if ext_data_pop_vars is None else ext_data_pop_vars

        n_inputs = len(ext_data_pop_vars)

        readout_neur_pop_vars = [] if readout_neur_pop_vars is None else readout_neur_pop_vars

        readout_syn_pop_vars = [] if readout_syn_pop_vars is None else readout_syn_pop_vars

        ext_data_input = np.ndarray((0)) if ext_data_input is None else ext_data_input
        ext_data_output = np.ndarray((0)) if ext_data_output is None else ext_data_output

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
            assert ext_dat.ndim == 2, \
                """Input data array does
                not have two dimensions."""

            _target_pop = self.neur_pops[target_pop]
            assert ext_dat.shape[1] == _target_pop.size, \
                """Input data size and
                input population size do not match."""

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
                (t_sign_readout.shape[0], self.neur_pops[readout_pop].size))

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

        _inp_pop = self.neur_pops["neur_input_input_pop"]

        _inp_pop.extra_global_params["u"].view[:
                                               ext_data_input.shape[0]] = ext_data_input
        _inp_pop.extra_global_params["t_sign"].view[:t_sign.shape[0]] = t_sign
        _inp_pop.extra_global_params["size_t_sign"].view[:] = t_sign.shape[0]

        _inp_pop.push_extra_global_param_to_device(
            "u", self.size_input * self.t_inp_max)
        _inp_pop.push_extra_global_param_to_device("t_sign", self.t_inp_max)

        # reset the input index counter before running the simulation
        _inp_pop.vars["idx_dat"].view[:] = 0
        _inp_pop.push_var_to_device("idx_dat")

        if self.plastic:
            _output_pop = self.neur_pops["neur_output_output_pop"]

            _output_pop.extra_global_params["u_trg"].view[:
                                                          ext_data_output.shape[0]] = ext_data_output
            _output_pop.extra_global_params["t_sign"].view[:t_sign.shape[0]] = t_sign
            _output_pop.extra_global_params["size_t_sign"].view[:] = t_sign.shape[0]

            _output_pop.push_extra_global_param_to_device(
                "u_trg", self.size_output * self.t_inp_max)
            _output_pop.push_extra_global_param_to_device(
                "t_sign", self.t_inp_max)

            _output_pop.vars["idx_dat"].view[:] = 0
            _output_pop.push_var_to_device("idx_dat")

        ####################################################

        for t in tqdm(range(T), disable=not show_progress):

            # manual variable manipulation

            self.push_ext_data(t, n_inputs, time_signatures_ext_data, ext_data_pop_vars, input_views, idx_data_heads)

            # check if validation run should be
            # executed in this time step.
            if data_validation and self.plastic:
                if (idx_validation_runs < t_sign_validation.shape[0]
                        and t_sign_validation[idx_validation_runs] == t):
                    results_validation.append(
                        self.run_validation(**data_validation[idx_validation_runs],
                                            show_progress=show_progress_val))

                    idx_validation_runs += 1

            self.genn_model.step_time()

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

    def run_validation(self, T, t_sign, ext_data_input, ext_data_pop_vars,
                       readout_neur_pop_vars, show_progress=True):
        '''
        This method runs a network simulation on the
        static twin after copying the current weight
        configuration to the twin. The output can
        be used e.g. to validate network performance.
        '''

        self.static_twin_net.genn_model.reinitialise()

        # store all the weights in the network
        weights = {}

        for key, syn_pop in self.syn_pops.items():
            view = syn_pop.vars["g"].view

            syn_pop.pull_var_from_device("g")

            # copy weights into new numpy array
            # (just storing a reference in the list could lead to
            # very unpredictable behavior)
            weights[key] = np.array(view)

        # transfer the copied weights to the temporary network
        for key, syn_pop in self.static_twin_net.syn_pops.items():
            view = syn_pop.vars["g"].view

            view[:] = weights[key]

            syn_pop.push_var_to_device("g")

        result_neur_arrays, _, result_spikes = self.static_twin_net.run_sim(
            T, t_sign, ext_data_input, np.ndarray(0),
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
                if time_signatures_ext_data[k][0] == t:
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

                if time_signatures_readout_neur_pop[k][0] == t:
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

                if time_signatures_readout_syn_pop[k][0] == t:
                    self.syn_pops[readout_pop].pull_var_from_device(
                        readout_var)
                    _dict_name = f'{readout_pop}_{readout_var}'

                    readout_syn_arrays[_dict_name][idx_readout_syn_pop_heads[k]] = np.reshape(
                        self.syn_pops[readout_pop].get_var_values(
                            readout_var),
                        readout_syn_arrays[_dict_name].shape[1:], order='F')

                    idx_readout_syn_pop_heads[k] += 1

                    time_signatures_readout_syn_pop[k] = time_signatures_readout_syn_pop[k][1:]