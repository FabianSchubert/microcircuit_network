## Run rate networks as event-based models in GeNN.

### Experiments
Experiments using the microcircuit architecture can be found in
`mc.experiments.mc`. Refer to the docstring of the respective
`run.py` files for details on running the models.

### GeNN Model Definitions
`mc.genn_models` contains subfolders with genn model definitions
of neurons and synapses. See `drop_in_synapses`as a
starting point when defining new neuron or synapse models.

### Network Architectures
`mc.network_architectures` contains definitions of synapses, layers
and networks that are all derived from base classes found in
`mc.network_base`.

#### Building and Running a Network Model

The `NetworkBase` class in `mc.network_base.network` is an abstract base
class, i.e. you need you need to define a child network class that inherits
from `NetworkBase` when defining a network model. In particular,
`NetworkBase` has an abstract method `setup` that has to be defined in your
child class, and this is where GeNN model definitions and network the architecture come together
into the final model.

To build your network in `setup`, you should
use the `LayerBase` and `SynapseBase` classes (or custom classes derived from
them) found in `mc.network_base.layer` and `mc.network_base.synapse`,
respectively. This is not strictly necessary, but it automates the extra steps involved
in setting up neuron populations and synapse populations for training and
testing, in particular adding custom updates to the model for weight updates.

**An example for building and running a model is given in `mc.min_example.py`**.
For simplicity, the construction of
neuron and synapse model definitions was done in the main script, but, as described
above, it is generally recommended to do so in an extra directory in `mc.genn_models`.
