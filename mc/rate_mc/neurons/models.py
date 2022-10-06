#! /usr/bin/env python3

from .model_defs import (pyr_model as pyr_model_def,
						int_model as int_model_def,
						input_model as input_model_def)

from pygenn.genn_model import create_custom_neuron_class

# Pyramidal Neuron
pyr_model = create_custom_neuron_class(**pyr_model_def)

# Interneuron                                            
int_model = create_custom_neuron_class(**int_model_def)

# Input Neuron
input_model = create_custom_neuron_class(**input_model_def)