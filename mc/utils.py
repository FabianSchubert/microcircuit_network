#! /usr/bin/env python3

from .defaults import default_synapse, default_neuron

from pygenn.genn_wrapper import NeuronModels, WeightUpdateModels, PostsynapticModels

from pygenn.genn_model import (create_custom_neuron_class, create_custom_weight_update_class,
								create_custom_postsynaptic_class)

import inspect

# Very bad style - get a list of the default model names by inspecting the class names
# in the respective models and excluding the Base, Custom and _SwigNonDynamicMeta classes.
default_neuron_models = [cl[0] for cl in inspect.getmembers(NeuronModels,inspect.isclass) 
                            if not cl[0] in ("Base","Custom","_SwigNonDynamicMeta")]
default_weight_update_models = [cl[0] for cl in inspect.getmembers(WeightUpdateModels,inspect.isclass)
                            if not cl[0] in ("Base","Custom","_SwigNonDynamicMeta")]
default_postsynaptic_models = [cl[0] for cl in inspect.getmembers(PostsynapticModels,inspect.isclass)
                            if not cl[0] in ("Base","Custom","_SwigNonDynamicMeta")]

def prepare_neur_model_dict(_neur_def):

	_neur_def_tmp = dict(default_neuron)

	for v,k in _neur_def:
        _neur_def_tmp[k] = v

    if(_neur_def_tmp["neuron"] not in default_neuron_models):
    	_neur_def_tmp["neuron"] = create_custom_neuron_class(**_neur_def_tmp["neuron"])

    return _neur_def_tmp

def prepare_syn_model_dict(_syn_def):

	_syn_def_tmp = dict(default_synapse)

	for v,k in _syn_def:
    	_syn_def_tmp[k] = v 

    # turn the dictionary defining the weight update model into an actual
    # GeNN weight update model class unless it is just a string referring 
    # to one of the default models.
    if(_syn_def_tmp["w_update_model"] not in default_weight_update_models):
        _syn_def_tmp["w_update_model"] = create_custom_weight_update_class(**_syn_def_tmp["w_update_model"])

    # the same with the postsynaptic model
    if(_syn_def_tmp["postsyn_model"] not in default_postsynaptic_models):
        
        _syn_def_tmp["postsyn_model"] = create_custom_postsynaptic_class(**_syn_def_tmp["postsyn_model"])

    return _syn_def_tmp


