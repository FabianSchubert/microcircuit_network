## Run rate network as event-based models in GeNN.

### Experiments
Experiments using the microcircuit architecture can be found in
`mc.experiments.mc`. Refer to the docstring of the respective
`run.py` files for details on running the models.

### GeNN Model Definitions
`mc.genn_models` contains subfolders with genn model definitions
of neurons and synapses. `drop_in_synapses` should be used as a
starting point when defining new neuron or synapse models.

### Network Architectures
`mc.network_architectures` contains definitions of synapses, layers
and networks that are all derived from base classes found in
`mc.network_base`.

### Building a Network Model
To create a network model for running an experiment/simulation,
GeNN model definitions and network architectures can be merged
into the final model. Unfortunately, this can not be done automatically
yet, and should be implemented in the `setup` function
of the network class. However, synapse and layer classes take neuron and
synapse defintions as module objects upon instantiation, which should
be imported from `mc.genn_models`.

