#! /usr/bin/env python3
"""
Definitions of the special layer classes.
"""

from .synapses import SynapseIP, SynapsePI

from pygenn.genn_model import create_custom_neuron_class

from network_base.layer import LayerBase

class HiddenLayer(LayerBase):

    def __init__(self, name, genn_model,
                 pyr_mod_dat, int_mod_dat,
                 synapseIP_def, synapsePI_def,
                 Npyr, Nint, plastic=True, read_only_weights=False,
                 optimizer_params={}):

        super().__init__(
            name, genn_model,
            plastic, read_only_weights)

        pyr_model_def = pyr_mod_dat["model_def"]
        pyr_param_space = pyr_mod_dat["param_space"]
        pyr_var_space = pyr_mod_dat["var_space"]
        _pyr_model = create_custom_neuron_class(**pyr_model_def)

        int_model_def = int_mod_dat["model_def"]
        int_param_space = int_mod_dat["param_space"]
        int_var_space = int_mod_dat["var_space"]
        _int_model = create_custom_neuron_class(**int_model_def)

        self.add_neur_pop("pyr_pop", Npyr, _pyr_model,
                          pyr_param_space, pyr_var_space,
                          optimizer_params=optimizer_params)
        self.add_neur_pop("int_pop", Nint, _int_model,
                          int_param_space, int_var_space,
                          optimizer_params=optimizer_params)

        _synapseIP = SynapseIP(Npyr, **synapseIP_def)
        _synapsePI = SynapsePI(Nint, **synapsePI_def)

        self.add_syn_pop("int_pop", "pyr_pop", _synapseIP,
                         optimizer_params=optimizer_params)
        self.add_syn_pop("pyr_pop", "int_pop", _synapsePI,
                         optimizer_params=optimizer_params)


class OutputLayer(LayerBase):

    def __init__(self, name, genn_model,
                 output_mod_dat, N,
                 plastic=True, read_only_weights=False,
                 optimizer_params={}):

        super().__init__(
            name, genn_model,
            plastic, read_only_weights)

        output_model_def = output_mod_dat["model_def"]
        output_param_space = output_mod_dat["param_space"]
        output_var_space = output_mod_dat["var_space"]

        _output_param_space = dict(output_param_space)

        _output_model = create_custom_neuron_class(**output_model_def)

        self.add_neur_pop("output_pop", N, _output_model,
                          _output_param_space, output_var_space,
                          optimizer_params=optimizer_params)


class InputLayer(LayerBase):

    def __init__(self, name, genn_model,
                 input_mod_dat, N,
                 plastic=True, read_only_weights=False,
                 optimizer_params={}):

        super().__init__(
            name, genn_model,
            plastic, read_only_weights)

        input_model_def = input_mod_dat["model_def"]
        input_param_space = input_mod_dat["param_space"]
        input_var_space = input_mod_dat["var_space"]

        _input_param_space = dict(input_param_space)
        #_input_param_space["pop_size"] = N

        _input_model = create_custom_neuron_class(**input_model_def)

        self.add_neur_pop("input_pop", N, _input_model,
                          _input_param_space, input_var_space,
                          optimizer_params=optimizer_params)



