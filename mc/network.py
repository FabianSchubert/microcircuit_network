#! /usr/bin/env python3

from pygenn.genn_model import (create_custom_neuron_class,
                                create_custom_weight_update_class,
                                create_custom_current_source_class,
                                create_custom_postsynaptic_class,
                                init_connectivity, init_var,
                               GeNNModel)

from .utils import prepare_neur_model_dict, prepare_syn_model_dict

from .defaults import (default_synapse,default_neuron,default_network_params)

import numpy as np

from tqdm import tqdm

#from dataclasses import dataclass



class NetworkLayer:

    def __init__(self,_genn_model,layer_name,**kwargs):
        self.name = layer_name

        self.neur_pops = {}

        self.syn_pops = {}

        self.genn_model = _genn_model

        _neuron_pop_defs = kwargs.get("neur_pops",[])

        for _neuron_pop_def in _neuron_pop_defs:
            self.add_neuron_pop(_neuron_pop_def)

        _synapse_pop_defs = kwargs.get("syn_pops",[])

        for _synapse_pop_def in _synapse_pop_defs:
            _source = _synapse_pop_def["source"]
            _target = _synapse_pop_def["target"]
            self.add_synapse_pop(_source,_target,_synapse_pop_def)

    def add_neuron_pop(self,_neur_def):

        _short_name, _neur_def = prepare_neur_model_dict(_neur_def,self)

        self.neur_pops[_short_name] = self.genn_model.add_neuron_population(
                    **_neur_def)

    def add_synapse_pop(self,_source_pop,_target_pop,_syn_def):

        _syn_def["source_layer"] = self.name
        _syn_def["source_pop"] = _source_pop
        _syn_def["target_layer"] = self.name
        _syn_def["target_pop"] = _target_pop

        _short_name, _syn_def = prepare_syn_model_dict(_syn_def)

        self.syn_pops[_short_name] = self.genn_model.add_synapse_population(
            **_syn_def) 



class Network:

    def __init__(self,params,**kwargs):
        # name of the network used for initializing the GeNNModel
        self.name = params.get("name",default_network_params["name"])

        # Get the dimension of the input layer.
        self.dim_input_layer = params.get("dim_input",default_network_params["dim_input"]);

        # Get the dimension of the output layer.
        self.dim_output_layer = params.get("dim_output",default_network_params["dim_output"]);

        # Get the dimension of the hidden layer(s) as an ordered
        # list, ordered in the direction of forward processing.
        self.dim_hidden_layers = params.get("dim_hidden",default_network_params["dim_hidden"])

        # Infer the number of hidden layers from the length
        # of the list "dimh" in params holding the dimensions
        # of each hidden layer.
        self.n_hidden_layers = len(self.dim_hidden_layers)

        # total number of layers = hidden layers plus input and output layer
        self.n_layers = 2 + self.n_hidden_layers

        # create the GeNNModel
        self.model = GeNNModel("float",self.name)

        # integration time step in ms
        self.INT_STEP = params.get("dt",default_network_params["dt"])
        self.model.dT = self.INT_STEP

        # required for only adding input from certain synapses (basal or apical)
        #self.synapse_model_to_basal = create_custom_postsynaptic_class(
        #    "synapse_model_to_basal",
        #    apply_input_code = "$(vb) += $(inSyn);"
        #)
        # see above
        #self.synapse_model_to_apical = create_custom_postsynaptic_class(
        #    "synapse_model_to_apical",
        #    apply_input_code = "$(va) += $(inSyn);"
        #)


        self.layers = []

        self.synapse_populations = []

        # Fetch all GeNN model parameters from the parameter
        # dictionary.
        self.fetch_genn_model_definitions(params)

        # Builds all involved GeNN models.
        self.build_genn_models()

        # Fetch all initial values for GeNN model parameters
        # and variables.
        self.fetch_genn_model_initializations(params)

        # Build neuron populations from the neuron models
        # and the initial values.
        self.build_neuron_populations()



        self.add_synapses()

        # Build the model
        self.model.build()

        # Load the model
        self.model.load()

    def add_network_layer(self):
        pass

    def add_synapse_pop(self,_source_layer_pop,_target_layer_pop,**kwargs):

        kwargs["source_layer"] = _source_layer_pop[0]
        kwargs["source_pop"] = _source_layer_pop[1]

        kwargs["target_layer"] = _target_layer_pop[0]
        kwargs["target_pop"] = _target_layer_pop[1]

        kwargs = prepare_syn_model_dict(kwargs)

        self.synapse_populations.append(
            self.model.add_synapse_population(**kwargs))






    def add_neuron_model(self,params):
        '''
        _name = params["name"]
        _params = params["params"]
        _vars = params["variables"]
        _sim_code = params["sim_code"]
        _threshold_condition_code = params["threshold_condition_code"]
        _reset_code = params["reset_code"]
        '''
        self.neuron_models[params["class_name"]] = create_custom_neuron_class(
            **params)

    def fetch_genn_model_definitions(self,_params):
        # Dictionary that holds GeNN neuron models.
        # Can be extended/updated by add_neuron_model()
        self.neuron_models = {}

        # Get the relevant data defining the excitatory neuron model parameters.
        self.exc_model_def = _params.get("exc_model_def",
            default_network_params["exc_model_def"])

        # Get the relevant data defining the interneuron model parameters.
        self.int_model_def = _params.get("int_model_def",
            default_network_params["int_model_def"])

        # Definition of the input layer neurons. By default, input neurons
        # simply interpret their external input current as their rate.
        self.input_model_def = _params.get("input_model_def",
            default_network_params["input_model_def"])

        # Get the relevant data defining the output model parameters.
        self.output_model_def = _params.get("output_model_def",
            default_network_params["output_model_def"])

        # Definitions of the weight update models.
        self.weight_update_model_exc_to_exc_fwd_def = _params.get(
            "weight_update_model_exc_to_exc_fwd_def",
            default_network_params["weight_update_model_exc_to_exc_fwd_def"]
            )

        self.weight_update_model_exc_to_exc_back_def = _params.get(
            "weight_update_model_exc_to_exc_back_def",
            default_network_params["weight_update_model_exc_to_exc_back_def"]
            )

        self.weight_update_model_exc_to_int_def = _params.get(
            "weight_update_model_exc_to_int_def",
            default_network_params["weight_update_model_exc_to_int_def"]
            )

        self.weight_update_model_int_to_exc_def = _params.get(
            "weight_update_model_int_to_exc_def",
            default_network_params["weight_update_model_int_to_exc_def"]
            )

        self.weight_update_model_exc_to_int_back_def = _params.get(
            "weight_update_model_exc_to_int_back_def",
            default_network_params["weight_update_model_exc_to_int_back_def"]
            )

        # fetch postsynaptic integration model definitions

        self.postsyn_model_exc_to_exc_fwd_def = _params.get(
            "postsyn_model_exc_to_exc_fwd_def",
            default_network_params["postsyn_model_exc_to_exc_fwd_def"])

        self.postsyn_model_exc_to_exc_back_def = _params.get(
            "postsyn_model_exc_to_exc_fwd_def",
            default_network_params["postsyn_model_exc_to_exc_fwd_def"])

        self.postsyn_model_exc_to_int_def = _params.get(
            "postsyn_model_exc_to_int_def",
            default_network_params["postsyn_model_exc_to_int_def"])

        self.postsyn_model_int_to_exc_def = _params.get(
            "postsyn_model_int_to_exc_def",
            default_network_params["postsyn_model_int_to_exc_def"])

        self.postsyn_model_exc_to_int_back_def = _params.get(
            "postsyn_model_exc_to_int_back_def",
            default_network_params["postsyn_model_exc_to_int_back_def"])

        

    def fetch_matrix_type(self,_params):

        self.mat_type_exc_to_exc_fwd_def = _params.get(
            "mat_type_exc_to_exc_fwd_def",
            default_network_params["mat_type_exc_to_exc_fwd_def"])

        self.mat_type_exc_to_exc_back_def = _params.get(
            "mat_type_exc_to_exc_back_def",
            default_network_params["mat_type_exc_to_exc_back_def"])

        self.mat_type_exc_to_int_def = _params.get(
            "mat_type_exc_to_int_def",
            default_network_params["mat_type_exc_to_int_def"])

        self.mat_type_int_to_exc_def = _params.get(
            "mat_type_int_to_exc_def",
            default_network_params["mat_type_int_to_exc_def"])

        self.mat_type_exc_to_int_back_def = _params.get(
            "mat_type_exc_to_int_back_def",
            default_network_params["mat_type_exc_to_int_back_def"])


    def fetch_genn_model_initializations(self,_params):
        # Get the initialization data for parameters and variables of
        # the excitatory neuron model.
        self.exc_model_init = _params.get("exc_model_init",
            default_network_params["exc_model_init"])

        # Get the initialization data for parameters and variables of
        # the interneuron model.
        self.int_model_init = _params.get("int_model_init",
            default_network_params["int_model_init"])

        # Get the initialization data for parameters and variables of
        # the input neuron model.
        self.input_model_init = _params.get("input_model_init",
            default_network_params["input_model_init"])

        # Get the initialization data for parameters and variables of
        # the interneuron model.
        self.output_model_init = _params.get("output_model_init",
            default_network_params["output_model_init"])

        self.syn_input_to_hidden_exc_init = _params.get(
            "synapse_input_to_hidden_exc_init",
            default_network_params["synapse_input_to_hidden_exc_init"])

        self.syn_hidden_exc_to_hidden_int_init = _params.get(
            "synapse_hidden_exc_to_hidden_int_init",
            default_network_params["synapse_hidden_exc_to_hidden_int_init"])

        self.syn_hidden_int_to_hidden_exc_init = _params.get(
            "synapse_hidden_int_to_hidden_exc_init",
            default_network_params["synapse_hidden_int_to_hidden_exc_init"])

        self.syn_hidden_exc_to_hidden_exc_fwd_init = _params.get(
            "synapse_hidden_exc_to_hidden_exc_fwd_init",
            default_network_params["synapse_hidden_exc_to_hidden_exc_fwd_init"])

        self.syn_hidden_exc_to_hidden_exc_back_init = _params.get(
            "synapse_hidden_exc_to_hidden_exc_back_init",
            default_network_params["synapse_hidden_exc_to_hidden_exc_back_init"])

        self.syn_hidden_exc_to_output_init = _params.get(
            "synapse_hidden_exc_to_output_init",
            default_network_params["synapse_hidden_exc_to_output_init"])

        self.syn_output_to_hidden_exc_init = _params.get(
            "synapse_output_to_hidden_exc_init",
            default_network_params["synapse_output_to_hidden_exc_init"])

        self.syn_hidden_exc_to_hidden_int_back_init = _params.get(
            "synapse_hidden_exc_to_hidden_int_back_init",
            default_network_params["synapse_hidden_exc_to_hidden_int_back_init"])

        self.syn_output_to_hidden_int_init = _params.get(
            "synapse_output_to_hidden_int_init",
            default_network_params["synapse_output_to_hidden_int_init"])

    # Instantiate all involved GeNN models from the model definitions.
    def build_genn_models(self):

        # Instantiate the neuron models.
        self.add_neuron_model(self.exc_model_def)
        self.add_neuron_model(self.int_model_def)
        self.add_neuron_model(self.input_model_def)
        self.add_neuron_model(self.output_model_def)

        # Instantiate weight update models.
        self.weight_update_model_exc_to_exc_fwd = create_custom_weight_update_class(
            **self.weight_update_model_exc_to_exc_fwd_def)

        self.weight_update_model_exc_to_exc_back = create_custom_weight_update_class(
            **self.weight_update_model_exc_to_exc_back_def)

        self.weight_update_model_exc_to_int = create_custom_weight_update_class(
            **self.weight_update_model_exc_to_int_def)

        self.weight_update_model_int_to_exc = create_custom_weight_update_class(
            **self.weight_update_model_int_to_exc_def)

        self.weight_update_model_exc_to_int_back = create_custom_weight_update_class(
            **self.weight_update_model_exc_to_int_back_def)


    def build_neuron_populations(self):

        # Instantiate input neuron population.
        self.input_pop = self.model.add_neuron_population("input_pop",
            self.dim_input_layer,
            self.neuron_models["input"],
            self.input_model_init["parameters"],
            self.input_model_init["variables"]
        )

        # Lists holding the hidden excitatory and interneuron populations
        self.hidden_exc_pop = []
        self.hidden_int_pop = []

        # Append hidden neuron populations to both lists.
        for k in range(self.n_hidden_layers):
            self.hidden_exc_pop.append(
                self.model.add_neuron_population("hidden_exc_pop"+str(k),
                    self.dim_hidden_layers[k],
                    self.neuron_models["exc"],
                    self.exc_model_init["parameters"],
                    self.exc_model_init["variables"]
                )
            )
            _n_hidden_int = self.dim_hidden_layers[k+1] if k < self.n_hidden_layers - 1 else self.dim_output_layer
            self.hidden_int_pop.append(
                self.model.add_neuron_population("hidden_int_pop"+str(k),
                    _n_hidden_int,
                    self.neuron_models["int"],
                    self.int_model_init["parameters"],
                    self.int_model_init["variables"]
                )
            )

        # Add output layer population
        self.output_pop = self.model.add_neuron_population("output_pop",
            self.dim_output_layer,
            self.neuron_models["output"],
            self.output_model_init["parameters"],
            self.output_model_init["variables"]
        )

    def add_synapses(self):
        self.syn_input_to_hidden_exc = self.model.add_synapse_population(
            "syn_input_to_hidden_exc",                  # name
            "DENSE_INDIVIDUALG",                    # matrix type
            0,                                      # synaptic delay
            self.input_pop, self.hidden_exc_pop[0], # presyn. pop., postsyn. pop.
            self.weight_update_model_exc_to_exc_fwd,               # weight update model
            self.syn_input_to_hidden_exc_init["parameters"],  # parameters init (not the weights!)
            self.syn_input_to_hidden_exc_init["variables"],    # variables init (with g being the weights)
            {},             # variable init for specific presyn. parameters
            {},             # variable init for specific postsyn. parameters
            "DeltaCurr",    # postsynaptic model: what happens to the InSyn current
            {},             # postsynaptic parameter space init
            {}              # postsynaptic variable space init
        )
        self.syn_input_to_hidden_exc.ps_target_var = "Isyn_vb" # redirect input to vb

        self.syn_hidden_exc_to_hidden_int = []
        self.syn_hidden_int_to_hidden_exc = []
        self.syn_hidden_exc_to_hidden_exc_fwd = []
        self.syn_hidden_exc_to_hidden_exc_back = []
        self.syn_hidden_exc_to_hidden_int_back = []

        for k in range(self.n_hidden_layers):
            self.syn_hidden_exc_to_hidden_int.append(
                self.model.add_synapse_population(
                    "syn_hidden_exc_to_hidden_int"+str(k),
                    "DENSE_INDIVIDUALG",
                    0,
                    self.hidden_exc_pop[k], self.hidden_int_pop[k],
                    self.weight_update_model_exc_to_int,
                    self.syn_hidden_exc_to_hidden_int_init["parameters"],
                    self.syn_hidden_exc_to_hidden_int_init["variables"],
                    {},
                    {},
                    "DeltaCurr",
                    {},
                    {}
                )
            )

            self.syn_hidden_int_to_hidden_exc.append(
                self.model.add_synapse_population(
                    "syn_hidden_int_to_hidden_exc"+str(k),
                    "DENSE_INDIVIDUALG",
                    0,
                    self.hidden_int_pop[k], self.hidden_exc_pop[k],
                    self.weight_update_model_int_to_exc,
                    self.syn_hidden_int_to_hidden_exc_init["parameters"],
                    self.syn_hidden_int_to_hidden_exc_init["variables"],
                    {},
                    {},
                    "DeltaCurr",
                    {},
                    {}
                )
            )
            self.syn_hidden_int_to_hidden_exc[-1].ps_target_var = "Isyn_va_int" # redirect input to va_int

            if(k < self.n_hidden_layers - 1):
                self.syn_hidden_exc_to_hidden_exc_fwd.append(
                    self.model.add_synapse_population(
                        "syn_hidden_exc_to_hidden_exc_fwd"+str(k),
                        "DENSE_INDIVIDUALG",
                        0,
                        self.hidden_exc_pop[k], self.hidden_exc_pop[k+1],
                        self.weight_update_model_exc_to_exc_fwd,
                        self.syn_hidden_exc_to_hidden_exc_fwd_init["parameters"],
                        self.syn_hidden_exc_to_hidden_exc_fwd_init["variables"],
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {}
                    )
                )
                self.syn_hidden_exc_to_hidden_exc_fwd[-1].ps_target_var = "Isyn_vb" # redirect input to vb

                self.syn_hidden_exc_to_hidden_exc_back.append(
                    self.model.add_synapse_population(
                        "syn_hidden_exc_to_hidden_exc_fwd"+str(k),
                        "DENSE_INDIVIDUALG",
                        0,
                        self.hidden_exc_pop[k+1], self.hidden_exc_pop[k],
                        self.weight_update_model_exc_to_exc_back,
                        self.syn_hidden_exc_to_hidden_exc_fwd_init["parameters"],
                        self.syn_hidden_exc_to_hidden_exc_fwd_init["variables"],
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {}
                    )
                )
                self.syn_hidden_exc_to_hidden_exc_back[-1].ps_target_var = "Isyn_va_exc" # redirect input to va

                self.syn_hidden_exc_to_hidden_int_back.append(
                    self.model.add_synapse_population(
                        "syn_hidden_exc_to_hidden_int_back"+str(k),
                        "SPARSE_GLOBALG",
                        0,
                        self.hidden_exc_pop[k+1], self.hidden_int_pop[k],
                        self.weight_update_model_exc_to_int_back,
                        self.syn_hidden_exc_to_hidden_int_back["parameters"],
                        self.syn_hidden_exc_to_hidden_int_back["variables"],
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                        init_connectivity("OneToOne",{})
                    )
                )
                self.syn_hidden_exc_to_hidden_int_back[-1].ps_target_var = "u_td"

        self.syn_hidden_exc_to_output = self.model.add_synapse_population(
            "syn_hidden_exc_to_output",
            "DENSE_INDIVIDUALG",
            0,
            self.hidden_exc_pop[self.n_hidden_layers-1], self.output_pop,
            self.weight_update_model_exc_to_exc_fwd,
            self.syn_hidden_exc_to_output_init["parameters"],
            self.syn_hidden_exc_to_output_init["variables"],
            {},
            {},
            "DeltaCurr",
            {},
            {}
        )

        self.syn_output_to_hidden_exc = self.model.add_synapse_population(
            "syn_output_to_hidden_exc",
            "DENSE_INDIVIDUALG",
            0,
            self.output_pop, self.hidden_exc_pop[self.n_hidden_layers-1],
            self.weight_update_model_exc_to_exc_back,
            self.syn_output_to_hidden_exc_init["parameters"],
            self.syn_output_to_hidden_exc_init["variables"],
            {},
            {},
            "DeltaCurr",
            {},
            {}
        )
        self.syn_output_to_hidden_exc.ps_target_var = "Isyn_va_exc" # redirection input to va

        self.syn_output_to_hidden_int = self.model.add_synapse_population(
            "syn_output_to_hidden_int",
            "SPARSE_GLOBALG",
            0,
            self.output_pop, self.hidden_int_pop[self.n_hidden_layers-1],
            self.weight_update_model_exc_to_int_back,
            self.syn_output_to_hidden_int_init["parameters"],
            self.syn_output_to_hidden_int_init["variables"],
            {},
            {},
            "DeltaCurr",
            {},
            {},
            init_connectivity("OneToOne",{})
        )
        self.syn_output_to_hidden_int.ps_target_var = "u_td"

    def run_network_top_down_input(self,input_data,output_nudging_data,
                                    record_neur_pop_vars = None,
                                    record_syn_pop_vars = None):

        # The input data needs two dimensions: time and the input layer space.
        assert input_data.ndim == 2

        # The dimensions of the input data on the second axis must
        # match the dimensions of the input layer.
        assert input_data.shape[1] == self.dim_input_layer

        # The same for the top-down nudging input and the output layer.
        assert output_nudging_data.ndim == 2
        assert output_nudging_data.shape[1] == self.dim_output_layer

        _t_run = input_data.shape[0]


        _neur_recordings = {}
        _neur_var_views = {}

        _syn_recordings = {}

        if(record_neur_pop_vars):
            # record_neur_pop_vars should be either None or
            # a list of tuples, each holding the name of the
            # neuron population and the name of the variable
            # that should be recorded.
            assert type(record_neur_pop_vars) is list
            _record_var_data = True

            for rec_pair in record_neur_pop_vars:
                # create a dictionary entry for each population to be recorded,
                # holding a 2d numpy array for recording. 
                _dim = self.model.neuron_populations[rec_pair[0]].size
                _key = f'{rec_pair[0]}_{rec_pair[1]}'
                _neur_recordings[_key] = np.ndarray((_t_run,_dim))

                # add a view for each of the populations
                _neur_var_views[_key] = self.model.neuron_populations[rec_pair[0]].vars[rec_pair[1]].view
        else:
            _record_var_data = False

        if(record_syn_pop_vars):
            # record_syn_pop_vars should either be None or
            # a list of tuples, each holding the name of the
            # neuron population and the name of the variable
            # that should be recorded.
            assert type(record_syn_pop_vars) is list
            _record_syn_data = True

            for rec_pair in record_syn_pop_vars:
                # create a dictionary entry for each synapse population to be recorded,

                # holding a 3d numpy arry for recording weights. The axes of the array
                # represent (time, postsynaptic layer dimension, presynaptic layer dimension ).
                _dim_pre = self.model.synapse_populations[rec_pair[0]].src.size

                _dim_post = self.model.synapse_populations[rec_pair[0]].trg.size

                _syn_recordings[f'{rec_pair[0]}_{rec_pair[1]}'] = np.ndarray((_t_run,_dim_post,_dim_pre))        

        else:
            _record_syn_data = False
        
        _input_layer_view = self.input_pop.vars["u"].view
        _output_layer_view = self.output_pop.vars["vtrg"].view

        for t in tqdm(range(_t_run)):
            
            # Set the input layer voltage view to the input data slice
            # and push the data to the device.
            _input_layer_view[:] = input_data[t]
            self.input_pop.push_var_to_device("u")

            # Set the output layer nudging voltage view to the output nudging
            # data slice and push the data to the device.
            _output_layer_view[:] = output_nudging_data[t]
            self.output_pop.push_var_to_device("vtrg")

            self.model.step_time()

            if(_record_var_data):
                for rec_pair in record_neur_pop_vars:
                    _key = f'{rec_pair[0]}_{rec_pair[1]}'
                    self.model.neuron_populations[rec_pair[0]].pull_var_from_device(rec_pair[1])
                    _neur_recordings[_key][t] = _neur_var_views[_key]

            if(_record_syn_data):
                for rec_pair in record_syn_pop_vars:
                    _key = f'{rec_pair[0]}_{rec_pair[1]}'
                    self.model.synapse_populations[rec_pair[0]].pull_var_from_device(rec_pair[1])
                    _syn_recordings[_key][t] = np.reshape(
                        self.model.synapse_populations[rec_pair[0]].get_var_values(rec_pair[1]),
                        (_syn_recordings[_key].shape[2],
                            _syn_recordings[_key].shape[1])
                    ).T


        return _neur_recordings, _syn_recordings
