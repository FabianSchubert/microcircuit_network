#! /usr/bin/env python3

from .layers import InputLayer, HiddenLayer, OutputLayer

from .synapses.models import (SynapsePPBasal, SynapsePPApical,
                              SynapseIPBack, SynapsePINP)

from dataclasses import dataclass

from pygenn.genn_model import GeNNModel

import numpy as np

from tqdm import tqdm


@dataclass
class Network:

    name: str
    size_input: int
    size_hidden: list
    size_output: int
    dt: float = 0.1
    plastic: bool = True

    def __post_init__(self):

        self.genn_model = GeNNModel("float", self.name)

        self.genn_model.dT = self.dt

        self.n_hidden_layers = len(self.size_hidden)

        self.layers = []
        self.layers.append(
            InputLayer("input",
                       self.genn_model,
                       self.size_input))

        # there must be at least one hidden layer.
        assert len(self.size_hidden) > 0

        for k in range(self.n_hidden_layers-1):
            _l_size = self.size_hidden[k]
            _l_next_size = self.size_hidden[k+1]
            self.layers.append(
                HiddenLayer(f'hidden{k}',
                            self.genn_model, _l_size, _l_next_size,
                            plastic=self.plastic))

        self.layers.append(
            HiddenLayer(f'hidden{self.n_hidden_layers-1}',
                        self.genn_model,
                        self.size_hidden[self.n_hidden_layers-1],
                        self.size_output,
                        plastic=self.plastic))

        self.layers.append(
            OutputLayer("output", self.genn_model, self.size_output))

        self.update_neur_pops()

        ##############################################
        # cross-layer synapse populations

        self.cross_layer_syn_pops = []

        _N_in = self.neur_pops["neur_input_input_pop"].size

        self.cross_layer_syn_pops.append(
            SynapsePINP(_N_in).connect_pops(
                'syn_input_input_pop_to_hidden0_pyr_pop',
                self.genn_model,
                self.neur_pops["neur_hidden0_pyr_pop"],
                self.neur_pops["neur_input_input_pop"],
                plastic=self.plastic
            )
        )

        for k in range(self.n_hidden_layers-1):
            _l = self.layers[k+1]
            _l_next = self.layers[k+2]

            _N_in = _l.neur_pops["pyr_pop"].size

            self.cross_layer_syn_pops.append(
                SynapsePPBasal(_N_in).connect_pops(
                    f'syn_{_l.name}_pyr_pop_to_{_l_next.name}_pyr_pop',
                    self.genn_model,
                    _l_next.neur_pops["pyr_pop"],
                    _l.neur_pops["pyr_pop"],
                    plastic=self.plastic
                )
            )

            _N_in = _l_next.neur_pops["pyr_pop"].size

            self.cross_layer_syn_pops.append(
                SynapsePPApical(_N_in).connect_pops(
                    f'syn_{_l_next.name}_pyr_pop_to_{_l.name}_pyr_pop',
                    self.genn_model,
                    _l.neur_pops["pyr_pop"],
                    _l_next.neur_pops["pyr_pop"],
                    plastic=self.plastic
                )
            )

            self.cross_layer_syn_pops.append(
                SynapseIPBack().connect_pops(
                    f'syn_{_l_next.name}_pyr_pop_to_{_l.name}_int_pop',
                    self.genn_model,
                    _l.neur_pops["int_pop"],
                    _l_next.neur_pops["pyr_pop"],
                    plastic=self.plastic
                )
            )

        _N_in = self.layers[-2].neur_pops["pyr_pop"].size

        self.cross_layer_syn_pops.append(
            SynapsePPBasal(_N_in).connect_pops(
                f'syn_{self.layers[-2].name}_pyr_pop_to_output_output_pop',
                self.genn_model,
                self.layers[-1].neur_pops["output_pop"],
                self.layers[-2].neur_pops["pyr_pop"],
                plastic=self.plastic
            )
        )

        _N_in = self.layers[-1].neur_pops["output_pop"].size

        self.cross_layer_syn_pops.append(
            SynapsePPApical(_N_in).connect_pops(
                f'syn_output_output_pop_to_{self.layers[-2].name}_pyr_pop',
                self.genn_model,
                self.layers[-2].neur_pops["pyr_pop"],
                self.layers[-1].neur_pops["output_pop"],
                plastic=self.plastic
            )
        )

        self.cross_layer_syn_pops.append(
            SynapseIPBack().connect_pops(
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

        self.genn_model.build()
        self.genn_model.load()

        # only if this instance is plastic, we create a static
        # twin of the network as a member variable of itself
        # (otherwise, calling create_static_twin() would lead
        # to an infinite recursive loop).
        if(self.plastic):
            self.create_static_twin()

    def update_neur_pops(self):

        self.neur_pops = {}
        for layer in self.layers:
            for pop in layer.neur_pops.values():
                self.neur_pops[pop.name] = pop

    def update_syn_pops(self):

        self.syn_pops = {}

        for layer in self.layers:
            for pop in layer.syn_pops.values():
                self.syn_pops[pop.name] = pop

        for pop in self.cross_layer_syn_pops:
            self.syn_pops[pop.name] = pop

    def create_static_twin(self):

        self.static_twin_net = Network(
            "static_twin", self.size_input, self.size_hidden,
            self.size_output, dt=self.dt, plastic=False)

    def run_sim(self, T, ext_data_pop_vars, readout_neur_pop_vars,
                readout_syn_pop_vars, data_validation=None):
        '''
        ext_data_pop_vars should be a list tuples of the form

        (numpy_data, numpy_times, target_pop, target_var).

        numpy_times is a list of time indices whose length
        should be the same as the first dimension of numpy_data.

        readout_neur_pop_vars & readout_syn_pop_vars should both be lists
        of tuples of the form

        (readout_pop, readout_var, numpy_times),

        where numpy_times is an array specifying the readout times.

        data_validation should be a list of the form

        [t_validation, T_run, ext_data_validation,
                readout_neur_pop_vars_validation],

        where t_validation is a numpy array with ordered time signatures
        used to trigger a validation run.

        T_run should be a list of runtimes for the validation runs.??,??,

        ext_data_validation should be a list of lists, each following
        the same structure as ext_data_pop_vars. ext_data_validation
        must have the same length as the size of the t_validation array,
        as each time signature corresponds to one entry in ext_data_validation.

        Likewise, readout_neur_pop_vars_validation should be a list of lists,
        each following the structure of readout_neur_pop_vars.
        '''

        input_views = []

        # List of indices storing the current "head" index from
        # which to load when the next time data is pushed to a population.
        idx_data_heads = []

        n_inputs = len(ext_data_pop_vars)

        # List for copies of time signatures.
        time_signatures = []

        if(data_validation):
            # Copy of the validation run time signatures
            t_validation = np.array(data_validation[0])

            T_run_validation = data_validation[1]

            ext_data_validation = data_validation[2]

            readout_neur_pop_vars_validation = data_validation[3]

            readout_syn_pop_vars_validation = data_validation[4]

            # List to store the results of the validation runs.
            results_validation = []

            # Counter for keeping track of the next validation run
            idx_validation_runs = 0

        for ext_dat, numpy_times, target_pop, target_var in ext_data_pop_vars:

            assert ext_dat.ndim == 2, """Input data array does
            not have two dimensions."""

            _target_pop = self.neur_pops[target_pop]
            assert ext_dat.shape[1] == _target_pop.size, """Input data size and
            input population size do not match."""

            assert len(
                numpy_times) > 0, """Error: Passed empty input
                data time span list."""

            assert np.all(numpy_times[:-1] < numpy_times[1:]), """Error: Input times are not correctly
            ordered"""

            assert numpy_times.shape[0] == ext_dat.shape[0], """Error: Size of input data does not match
            number of time signatures."""

            time_signatures.append(np.array(numpy_times))

            idx_data_heads.append(0)

            input_views.append(_target_pop.vars[target_var].view)

        readout_views = {}

        readout_neur_arrays = {}

        readout_syn_arrays = {}

        time_signatures_readout_neur_pop = []
        idx_readout_neur_pop_heads = []

        time_signatures_readout_syn_pop = []
        idx_readout_syn_pop_heads = []

        for readout_pop, readout_var, t_sign in readout_neur_pop_vars:
            _dict_name = f'{readout_pop}_{readout_var}'
            readout_views[_dict_name] = self.neur_pops[readout_pop].vars[readout_var].view
            readout_neur_arrays[_dict_name] = np.ndarray(
                (t_sign.shape[0], self.neur_pops[readout_pop].size))

            time_signatures_readout_neur_pop.append(np.array(t_sign))
            idx_readout_neur_pop_heads.append(0)

        for readout_pop, readout_var, t_sign in readout_syn_pop_vars:
            _dict_name = f'{readout_pop}_{readout_var}'
            readout_syn_arrays[_dict_name] = np.ndarray((t_sign.shape[0],
                                                         self.syn_pops[readout_pop].trg.size,
                                                         self.syn_pops[readout_pop].src.size))

            time_signatures_readout_syn_pop.append(np.array(t_sign))
            idx_readout_syn_pop_heads.append(0)

        for t in tqdm(range(T)):

            for k in range(n_inputs):

                # if the array of time signatures is not empty...
                if(time_signatures[k].shape[0] > 0):

                    # check if the current time is equal to the
                    # current first element in the
                    # array of time signatures.
                    if(time_signatures[k][0] == t):
                        input_views[k][:] = ext_data_pop_vars[k][0][idx_data_heads[k]]

                        self.neur_pops[ext_data_pop_vars[k][2]].push_var_to_device(
                            ext_data_pop_vars[k][3])

                        idx_data_heads[k] += 1

                        # remove the current first element of the
                        # time signatures after use.
                        time_signatures[k] = time_signatures[k][1:]

            if(data_validation):
                if(idx_validation_runs < t_validation.shape[0]
                        and t_validation[idx_validation_runs] == t):

                    results_validation.append(
                        self.run_validation(T_run_validation[idx_validation_runs],
                                            ext_data_validation[idx_validation_runs],
                                            readout_neur_pop_vars_validation[idx_validation_runs],
                                            readout_syn_pop_vars_validation[idx_validation_runs]))

                    idx_validation_runs += 1

            self.genn_model.step_time()

            for k, (readout_pop, readout_var, _) in enumerate(readout_neur_pop_vars):

                if(time_signatures_readout_neur_pop[k].shape[0] > 0):

                    if(time_signatures_readout_neur_pop[k][0] == t):

                        self.neur_pops[readout_pop].pull_var_from_device(
                            readout_var)
                        _dict_name = f'{readout_pop}_{readout_var}'
                        readout_neur_arrays[_dict_name][idx_readout_neur_pop_heads[k]
                                                        ] = readout_views[_dict_name]

                        idx_readout_neur_pop_heads[k] += 1

                        time_signatures_readout_neur_pop[k] = time_signatures_readout_neur_pop[k][1:]

            for k, (readout_pop, readout_var, _) in enumerate(readout_syn_pop_vars):

                if(time_signatures_readout_syn_pop[k].shape[0] > 0):

                    if(time_signatures_readout_syn_pop[k][0] == t):

                        self.syn_pops[readout_pop].pull_var_from_device(
                            readout_var)
                        _dict_name = f'{readout_pop}_{readout_var}'

                        readout_syn_arrays[_dict_name][idx_readout_syn_pop_heads[k]] = np.reshape(
                            self.syn_pops[readout_pop].get_var_values(
                                readout_var),
                            readout_syn_arrays[_dict_name].shape[1:], order='F')

                        idx_readout_syn_pop_heads[k] += 1

                        time_signatures_readout_syn_pop[k] = time_signatures_readout_syn_pop[k][1:]

        if(data_validation):
            return readout_neur_arrays, readout_syn_arrays, results_validation
        else:
            return readout_neur_arrays, readout_syn_arrays

    def run_validation(self, T, ext_data_pop_vars,
                       readout_neur_pop_vars, readout_syn_pop_vars):
        '''
        This method runs a network simulation on the
        static twin after copying the current weight
        configuration to the twin. The output can
        be used e.g. to validate network performance.
        '''

        # store all the weights in the network
        weights = {}

        for key, syn_pop in self.syn_pops.items():

            # 36 standing for SPARSE_GLOBALG:
            # You can not read weights from this matrix type, and it is not
            # plastic anyway.
            if(syn_pop.matrix_type != 36):

                syn_pop.pull_var_from_device("g")

                weights[key] = np.array(syn_pop.get_var_values("g"))

        # transfer the copied weights to the temporary network

        for key, syn_pop in self.static_twin_net.syn_pops.items():

            if(syn_pop.matrix_type != 36):

                view = syn_pop.vars["g"].view
                view[:] = weights[key]

                syn_pop.push_var_to_device("g")

        result_neur_arrays, result_syn_arrays = self.static_twin_net.run_sim(
            T, ext_data_pop_vars, readout_neur_pop_vars, readout_syn_pop_vars)

        return result_neur_arrays, result_syn_arrays
