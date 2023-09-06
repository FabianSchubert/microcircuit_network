import numpy as np
import matplotlib.pyplot as plt

from network_base.network import NetworkBase
from network_base.layer import LayerBase
from network_base.synapse import SynapseBase

from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY
from pygenn.genn_model import create_custom_neuron_class

from genn_models.utils import (WU_TRANSMIT, WU_VAR_SPACE_TRANSMIT, WU_PARAM_SPACE_TRANSMIT,
                                  PS_TRANSMIT, PS_VAR_SPACE_TRANSMIT, PS_PARAM_SPACE_TRANSMIT,
                                  generate_plast_wu_dict, convert_neuron_mod_data_cont_to_event,
                                  WU_PARAM_SPACE_PLAST, WU_VAR_SPACE_PLAST)

from pygenn.genn_model import init_var

# choose the model type to be either "cont" or "event"
mod_type = "cont"

# the **continuous** neuron model definition
_neur_model_def = {
    "class_name": "neur_model",
    "param_names": [],
    "var_name_types": [("v", "scalar"),
                       ("r", "scalar"),
                       ("b", "scalar", VarAccess_READ_ONLY),
                       ("db", "scalar")],
    "sim_code": """
        $(v) = $(v) + DT * ($(b) + $(Isyn) - $(v));
        $(r) = tanh($(v));
    """
}
# parameter and variable initialisations
_neur_param_init = {}
_neur_var_init = {"v": 0.5, "r": 0.0, "b": 0.0, "db": 0.0}

# pack the previous information into a dictionary
neur_mod_dat = {
    "model_def": _neur_model_def,
    "param_space": _neur_param_init,
    "var_space": _neur_var_init
}

"""
If the model type is event, we convert
the continuous neuron model data into
a correspnding event-based model.

post_plast_vars must hold a list of postsynaptic
variable names used in plasticity rules present
in synapses projecting onto a population that uses
this neuron model.

th defines the event threshold.
"""
if mod_type == "event":
    neur_mod_dat = convert_neuron_mod_data_cont_to_event(neur_mod_dat, post_plast_vars=[], th=1e-3)

# create the genn neuron class
neur_model = create_custom_neuron_class(**neur_mod_dat["model_def"])

### Synapse Definitions

# use the predefined synaptic transmission model / parameters / variables
# depending on the model type.
w_update_model_transmit = dict(WU_TRANSMIT[mod_type])
wu_param_space_transmit = dict(WU_PARAM_SPACE_TRANSMIT[mod_type])
wu_var_space_transmit = dict(WU_VAR_SPACE_TRANSMIT[mod_type])

# initialiser for the weights.
WEIGHT_SCALE = 0.9
wu_var_space_transmit["g"] = init_var("Uniform", {"min": -WEIGHT_SCALE*np.sqrt(3.), "max": WEIGHT_SCALE*np.sqrt(3.)})

# no plasticity rule
f = None

# generate the plasticity model definition depending on the
# model type and the provided plasticity rule (empty in this case)
w_update_model_plast = generate_plast_wu_dict(mod_type, "weight_update_model", f)
wu_param_space_plast = dict(WU_PARAM_SPACE_PLAST[mod_type])
wu_var_space_plast = dict(WU_VAR_SPACE_PLAST[mod_type])

# use the predefined postsynaptic model / parameters/ variables
# depending on the model type
ps_model_transmit = dict(PS_TRANSMIT[mod_type])
ps_param_space_transmit = dict(PS_PARAM_SPACE_TRANSMIT[mod_type])
ps_var_space_transmit = dict(PS_VAR_SPACE_TRANSMIT[mod_type])

# pack all synapse data
syn_mod_dat = {
    "w_update_model_transmit": w_update_model_transmit,
    "w_update_model_plast": w_update_model_plast,
    "wu_param_space_transmit": wu_param_space_transmit,
    "wu_param_space_plast": wu_param_space_plast,
    "wu_var_space_transmit": wu_var_space_transmit,
    "wu_var_space_plast": wu_var_space_plast,
    "ps_model_transmit": ps_model_transmit,
    "ps_param_space_transmit": ps_param_space_transmit,
    "ps_var_space_transmit": ps_var_space_transmit,
    "norm_after_init": "sqrt"
}

class MyNetwork(NetworkBase):
    # overwrite the abstract setup function
    def setup(self, N):
        # create an empty layer
        self.layer = LayerBase("layer1", self.genn_model)
        # add a neuron population to the layer,
        # using the neuron model defined above.
        self.layer.add_neur_pop("neur_pop1", N,
                                neur_model, neur_mod_dat["param_space"], neur_mod_dat["var_space"])

        # create a synapse object with dense connectivity and zero delay.
        # This is an intermediate object that is not associated with a particular
        # synapse population, but can be used to connect populations
        # ("generate" synapse populations) using the object's synapse model.
        _syn = SynapseBase("DENSE_INDIVIDUALG", # connection type,
                           0, #delay
                           **syn_mod_dat) # synapse model data

        # We create a synapse population within the layer.
        # In this case, we connect "neur_pop1" to itself.
        self.layer.add_syn_pop("neur_pop1", # target population
                               "neur_pop1", # source population
                               _syn) # the synapse object for creating the connection

N = 1000 # numner of neurons

DT = 0.1 # time step
N_BATCHES = 1 # number of batches during simulation
N_BATCHES_VAL = 1 # number of batches during validation / testing (not used here)
SPIKE_BUFFER_SIZE = 0 # number of time steps in the spike buffer
SPIKE_BUFFER_SIZE_VAL = 0 # number of time steps in the validation / testing spike buffer
SPIKE_REC_POPS = [] # populations to record spikes from during simulation
SPIKE_REC_POPS_VAL = [] # populations to record spikes from during validation / testing

# create network instance
net = MyNetwork("mynet",
        DT, N_BATCHES, N_BATCHES_VAL,
        SPIKE_BUFFER_SIZE, SPIKE_BUFFER_SIZE_VAL,
        SPIKE_REC_POPS, SPIKE_REC_POPS_VAL, N)

# additional simulation parameters
sim_params = {
    # run time
    "T": 1000.,
    # list of external data that should be pushed to neuron variables at a given point in time (see run_sim docstring)
    "ext_data_pop_vars": [],
    # list of tuples defining from which population to record which variable, and when.
    # note that the complete name of a specific population is "neur_<layer_name>_<population name given inside layer>"
    "readout_neur_pop_vars": [("neur_layer1_neur_pop1", "r", np.linspace(0.,100.,int(100./DT)))],
    # same for recording synapse variables.
    "readout_syn_pop_vars": []
}

# run the simulation and get the results (the latter two are empty)
readout_neur_arrays, readout_syn_arrays, readout_spikes = net.run_sim(**sim_params)

# the key vor a recording is "<full population name>_<variable name>"
r_rec = readout_neur_arrays["neur_layer1_neur_pop1_r"]

plt.plot(r_rec[:,0])
plt.show()

import pdb
pdb.set_trace()
