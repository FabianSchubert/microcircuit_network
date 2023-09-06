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

mod_type = "event"

neur_model_def = {
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
neur_param_init = {}
neur_var_init = {"v": 0.5, "r": 0.0, "b": 0.0, "db": 0.0}

neur_mod_dat = {
    "model_def": neur_model_def,
    "param_space": neur_param_init,
    "var_space": neur_var_init
}

if mod_type == "event":
    neur_mod_dat = convert_neuron_mod_data_cont_to_event(neur_mod_dat, post_plast_vars=[], th=0.1)

neur_model = create_custom_neuron_class(**neur_mod_dat["model_def"])

### Synapse Definitions

w_update_model_transmit = dict(WU_TRANSMIT[mod_type])
wu_param_space_transmit = dict(WU_PARAM_SPACE_TRANSMIT[mod_type])
wu_var_space_transmit = dict(WU_VAR_SPACE_TRANSMIT[mod_type])

WEIGHT_SCALE = 2.5
wu_var_space_transmit["g"] = init_var("Uniform", {"min": -WEIGHT_SCALE, "max": WEIGHT_SCALE})

f = None

w_update_model_plast = generate_plast_wu_dict(mod_type, "weight_update_model", f)
wu_param_space_plast = dict(WU_PARAM_SPACE_PLAST[mod_type])
wu_var_space_plast = dict(WU_VAR_SPACE_PLAST[mod_type])

ps_model_transmit = dict(PS_TRANSMIT[mod_type])
ps_param_space_transmit = dict(PS_PARAM_SPACE_TRANSMIT[mod_type])
ps_var_space_transmit = dict(PS_VAR_SPACE_TRANSMIT[mod_type])

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

    def setup(self, N):

        self.layer = LayerBase("layer1", self.genn_model)
        self.layer.add_neur_pop("neur_pop1", N,
                                neur_model, neur_mod_dat["param_space"], neur_mod_dat["var_space"])
        _syn = SynapseBase("DENSE_INDIVIDUALG", 0, **syn_mod_dat)
        self.layer.add_syn_pop("neur_pop1", "neur_pop1", _syn)

N = 200

DT = 0.1
N_BATCHES = 1
N_BATCHES_VAL = 1
SPIKE_BUFFER_SIZE = 0
SPIKE_BUFFER_SIZE_VAL = 0
SPIKE_REC_POPS = []
SPIKE_REC_POPS_VAL = []

net = MyNetwork("mynet",
        DT, N_BATCHES, N_BATCHES_VAL,
        SPIKE_BUFFER_SIZE, SPIKE_BUFFER_SIZE_VAL,
        SPIKE_REC_POPS, SPIKE_REC_POPS_VAL, N)

sim_params = {
    "T": 100.,
    "ext_data_pop_vars": [],
    "readout_neur_pop_vars": [("neur_layer1_neur_pop1", "r_event", np.linspace(0.,100.,int(100./DT)))],
    "readout_syn_pop_vars": []
}

readout_neur_arrays, _, _ = net.run_sim(**sim_params)

r_rec = readout_neur_arrays["neur_layer1_neur_pop1_r_event"]

plt.plot(r_rec[:,0])
plt.show()

import pdb
pdb.set_trace()
