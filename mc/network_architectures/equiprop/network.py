import types
from dataclasses import InitVar

from .layers import (HiddenLayer, InputLayer, OutputLayer)

from .synapses import GenericSynapse

import typing

from network_base.network import NetworkBase

import numpy as np


class EquipropNetwork(NetworkBase):

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

        ### add layers sequentially

        self.layers = []

        self.layers.append(
                InputLayer(
                    "input",
                    self.genn_model,
                    self.model_def.neurons.input.mod_dat,
                    self.size_input,
                    plastic=self.plastic,
                    optimizer_params=self.optimizer_params))

        for k in range(self.n_hidden_layers):
            _l_size = self.size_hidden[k]

            self.layers.append(
                HiddenLayer(
                    f'hidden{k}',
                    self.genn_model,
                    self.model_def.neurons.hidden.mod_dat,
                    _l_size,
                    plastic=self.plastic,
                    optimizer_params=self.optimizer_params))

        self.layers.append(
                OutputLayer(
                    "output",
                    self.genn_model,
                    self.model_def.neurons.output.mod_dat,
                    self.size_output,
                    plastic=self.plastic,
                    optimizer_params=self.optimizer_params))

        #### add cross-layer synapse populations.

        self.cross_layer_syn_pops = []

        self.cross_layer_syn_pops.append(
            GenericSynapse(
                        **self.model_def.synapses.generic.mod_dat
                        ).connect_pops(
                'syn_input_input_pop_to_hidden0_hidden_pop',
                self.genn_model,
                self.neur_pops["neur_hidden0_hidden_pop"],
                self.neur_pops["neur_input_input_pop"],
                plastic=self.plastic,
                optimizer_params=self.optimizer_params
            )
        )

        for k in range(self.n_hidden_layers-1):

            _l = self.layers[k + 1]
            _l_next = self.layers[k + 2]

            self.cross_layer_syn_pops.append(
                GenericSynapse(
                               **self.model_def.synapses.generic.mod_dat
                               ).connect_pops(
                    f'syn_{_l.name}_hidden_pop_to_{_l_next.name}_hidden_pop',
                    self.genn_model,
                    _l_next.neur_pops["hidden_pop"],
                    _l.neur_pops["hidden_pop"],
                    plastic=self.plastic,
                    optimizer_params=self.optimizer_params
                )
            )

            self.cross_layer_syn_pops.append(
                GenericSynapse(
                               **self.model_def.synapses.generic.mod_dat
                               ).connect_pops(
                    f'syn_{_l_next.name}_hidden_pop_to_{_l.name}_hidden_pop',
                    self.genn_model,
                    _l.neur_pops["hidden_pop"],
                    _l_next.neur_pops["hidden_pop"],
                    plastic=self.plastic,
                    optimizer_params=self.optimizer_params
                )
            )

        self.cross_layer_syn_pops.append(
            GenericSynapse(
                           **self.model_def.synapses.generic.mod_dat
                           ).connect_pops(
                f'syn_{self.layers[-2].name}_hidden_pop_to_output_output_pop',
                self.genn_model,
                self.layers[-1].neur_pops["output_pop"],
                self.layers[-2].neur_pops["hidden_pop"],
                plastic=self.plastic,
                optimizer_params=self.optimizer_params
            )
        )

        self.cross_layer_syn_pops.append(
            GenericSynapse(
                           **self.model_def.synapses.generic.mod_dat
                           ).connect_pops(
                f'syn_output_output_pop_to_{self.layers[-2].name}_hidden_pop',
                self.genn_model,
                self.layers[-2].neur_pops["hidden_pop"],
                self.layers[-1].neur_pops["output_pop"],
                plastic=self.plastic,
                optimizer_params=self.optimizer_params
            )
        )

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

    def align_fb_weights(self):

        for _l in self.layers:

            if type(_l) == HiddenLayer:

                # really bad approach - skip the "hidden" part
                # the layer name, which is always "hidden<idx>"
                idx = int(_l.name[6:])

                _name_this_layer = _l.name

                if idx < (self.n_hidden_layers - 1):
                    _name_next_layer = f'hidden{idx + 1}'
                    _name_next_pop = "hidden_pop"
                else:
                    _name_next_layer = "output"
                    _name_next_pop = "output_pop"

                _name_syn_fwd = f'syn_{_name_this_layer}_hidden_pop_to_{_name_next_layer}_{_name_next_pop}'
                _syn_fwd = self.syn_pops[_name_syn_fwd]
                _synview_syn_fwd = _syn_fwd.vars["g"].view
                _syn_fwd.pull_var_from_device("g")

                _w_syn_fwd = np.reshape(np.array(_synview_syn_fwd), (_syn_fwd.src.size, _syn_fwd.trg.size))

                _name_syn_back = f'syn_{_name_next_layer}_{_name_next_pop}_to_{_name_this_layer}_hidden_pop'
                _syn_back = self.syn_pops[_name_syn_back]
                _synview_syn_back = _syn_back.vars["g"].view

                _synview_syn_back[:] = _w_syn_fwd.T.flatten()
                _syn_back.push_var_to_device("g")

    def custom_sim_step(self, l):

        if self.plastic and (l["t"] % l["NT_skip_batch_plast"] == 0):
            self.align_fb_weights()
