## Structure of the Parameter Dict that is passed to the network constructor

- {
	- "name": 
	name of the network, 

	- "layers": 
	[
		- {
			
			- "layer_name": 
			name of the first layer. Should be a 
			unique name among all layers in the model.,
				
			- "neur_pops": 
			[
				- {
					- "pop_name": 
					name of the neuron population in 
					the layer. This should only be a short name that is 
					only required to be unique WITHIN the layer.,

					- "num_neurons": 
					number of neurons within the 
					population,

					- "neuron": 
					Either a string referring to one of the
					default neuron models, or a dict with keys defining
					the parameters of a custom neuron model.,

					- "param_space": 
					dict with parameter names and values 
					of the neuron model.,

					- "var_space": 
					dict with variable names and initial 
					values.
				},

				- { second neuron population etc... },
				
				- ...
			],

			- "syn_pops": 
			[
				- {
					- "source": 
					name of the source population WITHIN the layer,

					- "target": 
					name of the target population WITHIN the layer,

					- a whole bunch of other dict elements defining the synapse
				},

				- {another synapse population within the layer},
				
				- ...
			]
		},

		- { another layer},

		- ...
	],

	- "cross_layer_syn_pops":
	[
		- {
			- "source_layer":
			name of the source layer,

			- "source_pop":
			name of the source population within
			the source layer (only use its identifier that is unique
			within this layer),

			- "target_layer":
			name of the target layer,

			- "target_pop":
			name of the target population within the
			target layer (only use its identifier that is unique
			within this layer),

			- a whole bunch of other dict elements defining the synapse
		}

		- {another cross-layer synapse population},

		- ...
	]
}

