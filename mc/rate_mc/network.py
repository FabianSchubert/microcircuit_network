#! /usr/bin/env python3

from .layers import InputLayer , HiddenLayer, OutputLayer

from .synapses.models import SynapsePPBasal, SynapsePPApical, SynapseIPBack, SynapsePINP

from copy import deepcopy

from dataclasses import dataclass

from pygenn.genn_model import GeNNModel

import numpy as np

from tqdm import tqdm

import ipdb

@dataclass
class Network:

	name: str
	size_input: int
	size_hidden: list
	size_output: int
	dt: float = 0.1

	def __post_init__(self):

		self.genn_model = GeNNModel("float",self.name)

		self.genn_model.dT = self.dt

		self.n_hidden_layers = len(self.size_hidden)

		self.layers = []
		self.layers.append(
			InputLayer("input",
				self.genn_model,self.size_input))

		# there must be at least one hidden layer.
		assert len(self.size_hidden) > 0

		for k in range(self.n_hidden_layers-1):
			_l_size = self.size_hidden[k]
			_l_next_size = self.size_hidden[k+1]
			self.layers.append(
				HiddenLayer(f'hidden{k}',
					self.genn_model,_l_size,_l_next_size))

		self.layers.append(
			HiddenLayer(f'hidden{self.n_hidden_layers-1}',
					self.genn_model,
					self.size_hidden[self.n_hidden_layers-1],
					self.size_output))

		self.layers.append(
			OutputLayer("output",self.genn_model,self.size_output))

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
				self.neur_pops["neur_input_input_pop"]
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
					_l.neur_pops["pyr_pop"]
				)
			)

			_N_in = _l_next.neur_pops["pyr_pop"].size

			self.cross_layer_syn_pops.append(
				SynapsePPApical(_N_in).connect_pops(
					f'syn_{_l_next.name}_pyr_pop_to_{_l.name}_pyr_pop',
					self.genn_model,
					_l.neur_pops["pyr_pop"],
					_l_next.neur_pops["pyr_pop"]
				)
			)



			self.cross_layer_syn_pops.append(
				SynapseIPBack().connect_pops(
					f'syn_{_l_next.name}_pyr_pop_to_{_l.name}_int_pop',
					self.genn_model,
					_l.neur_pops["int_pop"],
					_l_next.neur_pops["pyr_pop"]
				)
			)

		_N_in = self.layers[-2].neur_pops["pyr_pop"].size

		self.cross_layer_syn_pops.append(
			SynapsePPBasal(_N_in).connect_pops(
				f'syn_{self.layers[-2].name}_pyr_pop_to_output_output_pop',
				self.genn_model,
				self.layers[-1].neur_pops["output_pop"],
				self.layers[-2].neur_pops["pyr_pop"]
			)			
		)

		_N_in = self.layers[-1].neur_pops["output_pop"].size

		self.cross_layer_syn_pops.append(
			SynapsePPApical(_N_in).connect_pops(
				f'syn_output_output_pop_to_{self.layers[-2].name}_pyr_pop',
				self.genn_model,
				self.layers[-2].neur_pops["pyr_pop"],
				self.layers[-1].neur_pops["output_pop"]
			)			
		)

		self.cross_layer_syn_pops.append(
			SynapseIPBack().connect_pops(
				f'syn_output_pyr_pop_to_{self.layers[-2].name}_int_pop',
				self.genn_model,
				self.layers[-2].neur_pops["int_pop"],
				self.layers[-1].neur_pops["output_pop"]
			)
		)


		self.update_syn_pops()

		#
		#################################################

		self.genn_model.build()
		self.genn_model.load()


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


	def run_sim(self,ext_data_pop_vars,readout_neur_pop_vars,readout_syn_pop_vars,T_skip=1):

		# ext_data should be a list tuples of the form:
		# (numpy_data, target_pop, target_var).
		# numpy_data should be a 2d array of size 
		# T x <size of the target population>.
		# target_pop and target_var are the names of
		# of the targeted population and variable.

		input_views = []

		T = None

		for target_data, target_pop, target_var in ext_data_pop_vars:


			assert target_data.ndim == 2, """Input data array does 
			not have two dimensions.""" 

			_target_pop = self.neur_pops[target_pop]
			assert target_data.shape[1] == _target_pop.size, """Input data size and 
			input population size do not match.""" 

			if(T is None):
				T = target_data.shape[0]
			else:
				assert target_data.shape[0] == T, """Input data time steps do not
				match with rest of input data.""" 

			input_views.append(_target_pop.vars[target_var].view)

		T_rec = int(T/T_skip)

		readout_views = {}

		readout_neur_arrays = {}

		readout_syn_arrays = {}

		for readout_pop, readout_var in readout_neur_pop_vars:
			_dict_name = f'{readout_pop}_{readout_var}'
			readout_views[_dict_name] = self.neur_pops[readout_pop].vars[readout_var].view
			readout_neur_arrays[_dict_name] = np.ndarray((T_rec,self.neur_pops[readout_pop].size))



		for readout_pop, readout_var in readout_syn_pop_vars:
			_dict_name = f'{readout_pop}_{readout_var}'
			readout_syn_arrays[_dict_name] = np.ndarray((T_rec,
											self.syn_pops[readout_pop].trg.size,
											self.syn_pops[readout_pop].src.size))

		

		for t in tqdm(range(T)):

			for k in range(len(input_views)):
				input_views[k][:] = ext_data_pop_vars[k][0][t]
				self.neur_pops[ext_data_pop_vars[k][1]].push_var_to_device(
						ext_data_pop_vars[k][2])

			self.genn_model.step_time()

			if t%T_skip == 0:
				t_rec = int(t/T_skip)
				
				for readout_pop, readout_var in readout_neur_pop_vars:
					self.neur_pops[readout_pop].pull_var_from_device(readout_var)
					_dict_name = f'{readout_pop}_{readout_var}'
					readout_neur_arrays[_dict_name][t_rec] = readout_views[_dict_name]

				for readout_pop, readout_var in readout_syn_pop_vars:
					self.syn_pops[readout_pop].pull_var_from_device(readout_var)
					_dict_name = f'{readout_pop}_{readout_var}'

					readout_syn_arrays[_dict_name][t_rec] = np.reshape(
                        self.syn_pops[readout_pop].get_var_values(readout_var),
                        readout_syn_arrays[_dict_name].shape[1:],order='F'
                    )

		return readout_neur_arrays, readout_syn_arrays




			









'''
def create_multi_layer_network_def(name,N_in,N_h,N_out):

	# N_in: Size of the input layer.
	# N_h: List of sizes [N_h_pyr0, N_h_pyr1,...]
	# that defindes the size of the pyramidal population
	# in each hidden layer. Note that the size of the
	# interneuron population is given by the size of
	# the pyramidal population in the next layer (or the
	# output layer for the last hidden layer).

	_input_layer = deepcopy(input_layer)
	_input_layer["neur_pops"]["pyr_pops"]["num_neurons"] = 

	network_model = {
		"name": name,

		"layers": [ 
		_input_layer,

		 ]
	}
'''