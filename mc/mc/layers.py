#! /usr/bin/env python3
"""
Definitions of the Layer base class and
special layer classes.
"""

from .synapses import SynapseIP, SynapsePI

from pygenn.genn_model import create_custom_neuron_class


'''
f'hidden{k}',
self.genn_model,
self.model_def.neurons.pyr.mod_dat,
self.model_def.neurons.int.mod_dat,
self.model_def.synapses.IP.mod_dat,
self.model_def.synapses.PI.mod_dat,
_l_size, _l_next_size,
plastic=self.plastic
'''

class LayerBase:

    def __init__(self, name, genn_model, plastic=True):
        self.name = name
        self.genn_model = genn_model
        self.neur_pops = {}
        self.syn_pops = {}
        self.plastic = plastic

    def add_neur_pop(self, pop_name, size, neur_model, param_init, var_init):
        _full_name = f'neur_{self.name}_{pop_name}'
        _new_pop = self.genn_model.add_neuron_population(_full_name, size,
                                                         neur_model,
                                                         param_init,
                                                         var_init)
        self.neur_pops[pop_name] = _new_pop

    def add_syn_pop(self, target, source, syn_model):

        self.syn_pops[f'{source}_to_{target}'] = syn_model.connect_pops(
            f'syn_{self.name}_{source}_to_{target}',
            self.genn_model, self.neur_pops[target], self.neur_pops[source],
            plastic=self.plastic)


class HiddenLayer(LayerBase):

    def __init__(self, *args, **kwargs):

        super().__init__(*args[:-6], **kwargs)

        pyr_model_def = args[-6]["model_def"]
        pyr_param_space = args[-6]["param_space"]
        pyr_var_space = args[-6]["var_space"]
        _pyr_model = create_custom_neuron_class(**pyr_model_def)

        int_model_def = args[-5]["model_def"]
        int_param_space = args[-5]["param_space"]
        int_var_space = args[-5]["var_space"]
        _int_model = create_custom_neuron_class(**int_model_def)

        synapseIP_def = args[-4]
        synapsePI_def = args[-3]

        Npyr = args[-2]
        Nint = args[-1]

        self.add_neur_pop("pyr_pop", Npyr, _pyr_model,
                          pyr_param_space, pyr_var_space)
        self.add_neur_pop("int_pop", Nint, _int_model,
                          int_param_space, int_var_space)

        _synapseIP = SynapseIP(Npyr, **synapseIP_def)
        _synapsePI = SynapsePI(Nint, **synapsePI_def)

        self.add_syn_pop("int_pop", "pyr_pop", _synapseIP)
        self.add_syn_pop("pyr_pop", "int_pop", _synapsePI)


class OutputLayer(LayerBase):

    def __init__(self, *args, **kwargs):

        super().__init__(*args[:-2], **kwargs)

        output_model_def = args[-2]["model_def"]
        output_param_space = args[-2]["param_space"]
        output_var_space = args[-2]["var_space"]

        N = args[-1]

        _output_param_space = dict(output_param_space)
        _output_param_space["pop_size"] = N

        _output_model = create_custom_neuron_class(**output_model_def)

        self.add_neur_pop("output_pop", N, _output_model,
                          _output_param_space, output_var_space)


class InputLayer(LayerBase):

    def __init__(self, *args, **kwargs):

        super().__init__(*args[:-2], **kwargs)

        input_model_def = args[-2]["model_def"]
        input_param_space = args[-2]["param_space"]
        input_var_space = args[-2]["var_space"]

        N = args[-1]

        _input_param_space = dict(input_param_space)
        _input_param_space["pop_size"] = N

        _input_model = create_custom_neuron_class(**input_model_def)

        self.add_neur_pop("input_pop", N, _input_model,
                          _input_param_space, input_var_space)
