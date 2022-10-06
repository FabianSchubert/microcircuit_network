#! /usr/bin/env python3

from .neurons.models import pyr_model, int_model, input_model

from .neurons.params import (pyr_hidden_param_space,
							pyr_output_param_space,
							int_param_space,
							input_param_space)

from .neurons.var_inits import pyr_var_space, int_var_space, input_var_space

'''
from .synapses.models import (synapse_pp_basal,
                                synapse_pp_apical,
                                synapse_pi,synapse_ip)
'''

from .synapses.models import (SynapsePPBasal,
                                SynapsePPApical,
                                SynapsePI,SynapseIP)


from dataclasses import dataclass

class LayerBase:

	def __init__(self,name,genn_model):
		self.name = name
		self.genn_model = genn_model
		self.neur_pops = {}
		self.syn_pops = {}

	def add_neur_pop(self,pop_name,size,neur_model,param_init,var_init):
		_full_name = f'neur_{self.name}_{pop_name}'
		self.neur_pops[pop_name] = self.genn_model.add_neuron_population(_full_name, size, 
									neur_model, param_init, var_init)

	def add_syn_pop(self,source,target,syn_model):

		self.syn_pops[f'{source}_to_{target}'] = syn_model.connect_pops(
			f'syn_{self.name}_{source}_to_{target}',
			self.genn_model,self.neur_pops[source],self.neur_pops[target])


class HiddenLayer(LayerBase):

	def __init__(self,*args,**kwargs):

		super().__init__(*args[:-2],**kwargs)
		
		Npyr = args[-2]
		Nint = args[-1]

		self.add_neur_pop("pyr_pop",Npyr,pyr_model,pyr_hidden_param_space,pyr_var_space)
		self.add_neur_pop("int_pop",Nint,int_model,int_param_space,int_var_space)

		self.add_syn_pop("pyr_pop","int_pop",SynapsePI(Npyr))
		self.add_syn_pop("int_pop","pyr_pop",SynapseIP(Nint))

class OutputLayer(LayerBase):

	def __init__(self,*args,**kwargs):

		super().__init__(*args[:-1],**kwargs)

		N = args[-1]

		self.add_neur_pop("pyr_pop",N,pyr_model,pyr_output_param_space,pyr_var_space)

class InputLayer(LayerBase):

	def __init__(self,*args,**kwargs):

		super().__init__(*args[:-1],**kwargs)

		N = args[-1]

		self.add_neur_pop("input_pop",N,input_model,input_param_space,input_var_space)



'''
input_layer = {
	"layer_name": "input_layer",

	"neur_pops": [
		(pyr_model | 
		{"pop_name": "pyr_pop",
		 "num_neurons": None # should be manually set
		})
	],

	"syn_pops": []
}

hidden_layer = {
	"layer_name": None, # should be manually set (e.g. hidden_layer0, hidden_layer1 etc.)
	"neur_pops": [
		(pyr_model |
		{"pop_name": "pyr_pop",
		"num_neurons": None # should be manually set
		}),
		(int_model |
		{"pop_name": "int_pop",
		 "num_neurons": None
		})
	],

	"syn_pops": [
		({"source": "pyr_pop",
			"target": "int_pop"} | synapse_pi),
		({"source": "int_pop",
			"target": "pyr_pop"} | synapse_ip)
	]
}

output_layer = {
	"layer_name": "output_layer",

	"neur_pops": [
		(pyr_model | 
		{"pop_name": "pyr_pop",
		 "num_neurons": None # should be manually set
		})
	],

	"syn_pops": []
}
'''