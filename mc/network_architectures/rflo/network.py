import types
from dataclasses import InitVar

from .layers import InputLayer, HiddenLayer, OutputLayer

from .synapses import SynapseHidden, SynapseHiddenIn, SynapseOutHidden, SynapseHiddenOut

import typing

from network_base.network import NetworkBase

import numpy as np

class RFLONetwork(NetworkBase):

    def setup(self, size_input: int,
              size_hidden: int,
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

        self.input_layer = InputLayer("input", self.genn_model,
                                      self.model_def.neurons.input.mod_dat,
                                      self.size_input, plastic=self.plastic,
                                      optimizer_params=self.optimizer_params)

        self.hidden_layer = HiddenLayer("hidden", self.genn_model,
                                     self.model_def.neurons.hidden.mod_dat,
                                     self.model_def.synapses.hidden.mod_dat,
                                     self.size_hidden, plastic=self.plastic,
                                     optimizer_params=self.optimizer_params)

        self.output_layer = OutputLayer("output", self.genn_model,
                                        self.model_def.neurons.output.mod_dat,
                                        self.size_output, plastic=self.plastic,
                                        optimizer_params=self.optimizer_params)

        self.syn_pop_hidden_in = SynapseHiddenIn(
                **self.model_def.synapses.hidden_in.mod_dat).connect_pops(
                        "syn_input_pop_to_hidden_pop",
                        self.genn_model,
                        self.neur_pops["neur_hidden_hidden_pop"],
                        self.neur_pops["neur_input_input_pop"],
                        plastic=self.plastic,
                        optimizer_params=self.optimizer_params)

        self.syn_pop_out_hidden = SynapseOutHidden(
                **self.model_def.synapses.out_hidden.mod_dat).connect_pops(
                        "syn_hidden_pop_to_output_pop",
                        self.genn_model,
                        self.neur_pops["neur_output_output_pop"],
                        self.neur_pops["neur_hidden_hidden_pop"],
                        plastic=self.plastic,
                        optimizer_params=self.optimizer_params)

        self.syn_pop_hidden_out = SynapseHiddenOut(
                **self.model_def.synapses.hidden_out.mod_dat).connect_pops(
                        "syn_out_pop_to_hidden_pop",
                        self.genn_model,
                        self.neur_pops["neur_hidden_hidden_pop"],
                        self.neur_pops["neur_output_output_pop"],
                        plastic=self.plastic,
                        optimizer_params=self.optimizer_params)

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
