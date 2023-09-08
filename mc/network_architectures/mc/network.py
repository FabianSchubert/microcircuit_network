"""
This module defines the network class used to
construct an instance of the dendritic microcircuit model.
"""

import types
from dataclasses import InitVar

from .layers import (HiddenLayer, InputLayer, OutputLayer)

from .synapses import (SynapseIPBack, SynapsePINP,
                       SynapsePPApical, SynapsePPBasal)

import typing

from network_base.network import NetworkBase

import numpy as np


class MCNetwork(NetworkBase):
    """
        model_def (module):

            A module object that contains a hierarchical
            structure of model definitions. The basic module
            structure must be:

                model
                ├── neurons
                │    ├── neuron_model1.py
                │    ├── neuron_model2.py
                │    ├──  ...
                │
                └── synapses
                     ├── synapse_model1.py
                     ├── synapse_model2.py
                     ├──  ...

            Each neuron_model.py must contain a dict mod_dat
            containing the genn neuron model definition as a dict
            as well as the appropriate dicts vor variable / parameter
            initialization. e.g.

                mod_dat = {
                    "model_def": model_def,
                    "param_space": param_space,
                    "var_space": var_space
                }

            For synapse_model.py, the minimal contents of mod_dat
            must be:

                mod_dat = {
                    "w_update_model_transmit": w_update_model_transmit,
                    "w_update_model_plast": w_update_model_plast,
                    "wu_param_space_transmit": wu_param_space_transmit,
                    "wu_param_space_plast": wu_param_space_plast,
                    "wu_var_space_transmit": wu_var_space_transmit,
                    "wu_var_space_plast": wu_var_space_plast,
                    "ps_model_transmit": ps_model_transmit,
                    "ps_param_space_transmit": ps_param_space_transmit,
                    "ps_var_space_transmit": ps_var_space_transmit,
                    "norm_after_init": "sqrt"
                }

            Note that the model definitions are split into a "transmit"
            and a "plast" part. If the model is run in "plastic" mode,
            both definitions are merged before the GeNN model is built,
            whereas for plastic = False, only the "transmit" part is
            used. If merging takes place, code snippets from the "plast"
            definition are placed after the corresponding "transmit"
            code snippets (if both are present).

            "norm_after_init" can be False, "lin", "lin_inv",
            "sqrt" or "sqrt_inv". For "lin" and "lin_inv", weights
            are multiplied with 1/N_pre or 1/N_post respectively
            during model initialization, where N_pre and N_post are
            the number neurons in the pre- or postsynaptic population.
            The same holds for "sqrt" and "sqrt_inv", except that
            the factors are 1/sqrt(N_pre) and 1/sqrt(N_post).

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

    def setup(self, size_input: int,
              size_hidden: typing.List[int],
              size_output: int,
              model_def: types.ModuleType,
              optimizer_params={},
              cs_in_init: InitVar[typing.Any] = None,
              cs_out_init: InitVar[typing.Any] = None,
              cs_in_init_static_twin: InitVar[typing.Any] = None,
              cs_out_init_static_twin: InitVar[typing.Any] = None):

        self.size_input = size_input
        self.size_hidden = size_hidden
        self.size_output = size_output
        self.model_def = model_def
        self.optimizer_params = optimizer_params

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
        #################################################

        ############################
        # current sources
        if cs_in_init:

            _cs_in_init = cs_in_init if self.plastic else cs_in_init_static_twin

            self.add_current_source(
                "cs_in",
                "neur_input_input_pop",
                _cs_in_init["model"],
                _cs_in_init["params"],
                _cs_in_init["vars"],
                _cs_in_init["extra_global_params"])
        else:
            self.cs_in = None

        if cs_out_init:

            _cs_out_init = cs_out_init if self.plastic else cs_out_init_static_twin

            self.add_current_source(
                "cs_out",
                "neur_output_output_pop",
                _cs_out_init["model"],
                _cs_out_init["params"],
                _cs_out_init["vars"],
                _cs_out_init["extra_global_params"])
        else:
            self.cs_out = None
        ############################

    def custom_sim_step(self, l,
                        force_fb_align=False,
                        force_self_pred_state=False):

        if self.plastic and (l["t"] % l["NT_skip_batch_plast"] == 0):
            if force_fb_align:
                self.align_fb_weights()
            if force_self_pred_state:
                self.init_self_pred_state()

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
