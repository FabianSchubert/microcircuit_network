#! /usr/bin/env python3

from .neurons.models import pyr_model, int_model, input_model, output_model

from .neurons.params import (pyr_hidden_param_space,
                             output_param_space,
                             int_param_space,
                             input_param_space)

from .neurons.var_inits import (pyr_var_space, int_var_space,
                                input_var_space, output_var_space)

'''
from .synapses.models import (synapse_pp_basal,
                                synapse_pp_apical,
                                synapse_pi,synapse_ip)
'''

from .synapses.models import SynapseIP, SynapsePI


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

        super().__init__(*args[:-2], **kwargs)

        Npyr = args[-2]
        Nint = args[-1]

        self.add_neur_pop("pyr_pop", Npyr, pyr_model,
                          pyr_hidden_param_space, pyr_var_space)
        self.add_neur_pop("int_pop", Nint, int_model,
                          int_param_space, int_var_space)

        self.add_syn_pop("int_pop", "pyr_pop", SynapseIP(Npyr))
        self.add_syn_pop("pyr_pop", "int_pop", SynapsePI(Nint))


class OutputLayer(LayerBase):

    def __init__(self, *args, **kwargs):

        super().__init__(*args[:-1], **kwargs)

        N = args[-1]

        _output_param_space = dict(output_param_space)
        _output_param_space["pop_size"] = N

        self.add_neur_pop("output_pop", N, output_model,
                          _output_param_space, output_var_space)


class InputLayer(LayerBase):

    def __init__(self, *args, **kwargs):

        super().__init__(*args[:-1], **kwargs)

        N = args[-1]

        _input_param_space = dict(input_param_space)
        _input_param_space["pop_size"] = N

        self.add_neur_pop("input_pop", N, input_model,
                          _input_param_space, input_var_space)
