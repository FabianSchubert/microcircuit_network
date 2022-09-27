#! /usr/bin/env python3

from .models import pyr_model, int_model
from .params import pyr_param_space, int_param_space
from .var_inits import pyr_var_space, int_var_space

# Pyramidal Neuron
pyr_neuron = {        
    "neuron": pyr_model,
    "param_space": pyr_param_space,
    "var_space": pyr_var_space
}
#

# Interneuron                                            
int_neuron = {
    "neuron": int_model,
    "param_space": int_param_space,
    "var_space": int_var_space
}