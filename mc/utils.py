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

def create_neur_pop_name(_layer_name,_pop_name):
    return f'neurpop_{_layer_name}_{_pop_name}'

def create_syn_pop_name(_source_layer_name,_source_pop_name,_target_layer_name,_target_pop_name):
    return (f'synpop_'
        +f'{_source_layer_name}_'
        +f'{_source_pop_name}_'
        +f'to_{_target_layer_name}_'
        +f'{_target_pop_name}')

def create_syn_pop_name_short(_source_pop_name, _target_pop_name):
    return (f'synpop_'
        +f'{_source_pop_name}_'
        +f'to_{_target_pop_name}')

def prepare_neur_model_dict(_neur_def,_layer):

    _neur_def_tmp = dict(default_neuron)

    for k,v in _neur_def.items():
        _neur_def_tmp[k] = v

    _pop_name_short = _neur_def_tmp["pop_name"]
    _neur_def_tmp["pop_name"] = create_neur_pop_name(_layer.layer_name,_neur_def["pop_name"])

    #_neur_def_tmp["pop_name"] = f'{_layer_name}_neurpop_{_neur_def_tmp["pop_name"]}'

    if(_neur_def_tmp["neuron"] not in default_neuron_models):
        _neur_def_tmp["neuron"] = create_custom_neuron_class(**_neur_def_tmp["neuron"])

    return _pop_name_short, _neur_def_tmp

def prepare_syn_model_dict(_syn_def):
    
    _syn_def_tmp = dict(default_synapse)

    for k,v in _syn_def.items():
        _syn_def_tmp[k] = v

    _pop_name_short = create_syn_pop_name_short(_syn_def_tmp["source_pop"],_syn_def_tmp["target_pop"])

    _syn_def_tmp["pop_name"] = create_syn_pop_name(_syn_def_tmp["source_layer"],_syn_def_tmp["source_pop"],
                                                    _syn_def_tmp["target_layer"],_syn_def_tmp["target_pop"])

    _syn_def_tmp["source"] = create_neur_pop_name(_syn_def_tmp["source_layer"],_syn_def_tmp["source_pop"])
    _syn_def_tmp["target"] = create_neur_pop_name(_syn_def_tmp["target_layer"],_syn_def_tmp["target_pop"])

    # turn the dictionary defining the weight update model into an actual
    # GeNN weight update model class unless it is just a string referring 
    # to one of the default models.
    if(_syn_def_tmp["w_update_model"] not in default_weight_update_models):
        _syn_def_tmp["w_update_model"] = create_custom_weight_update_class(**_syn_def_tmp["w_update_model"])

    # the same with the postsynaptic model
    if(_syn_def_tmp["postsyn_model"] not in default_postsynaptic_models):
        
        _syn_def_tmp["postsyn_model"] = create_custom_postsynaptic_class(**_syn_def_tmp["postsyn_model"])

    _syn_def_tmp.pop("source_layer")
    _syn_def_tmp.pop("source_pop")
    _syn_def_tmp.pop("target_layer")
    _syn_def_tmp.pop("target_pop")

    _ps_target_var = _syn_def_tmp["ps_target_var"]
    _syn_def_tmp.pop("ps_target_var")


    return _pop_name_short, _ps_target_var, _syn_def_tmp


