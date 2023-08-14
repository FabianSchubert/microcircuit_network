#! /usr/bin/env python3

from pygenn.genn_wrapper.Models import VarAccess_REDUCE_BATCH_SUM, VarAccessMode_READ_ONLY

from pygenn.genn_model import create_custom_custom_update_class

act_func = lambda x: f'log(1.0+exp(1.0*({x})))'
# act_func = lambda x: f'tanh({x})'
# act_func = lambda x: f'{x}'

param_change_batch_reduce = create_custom_custom_update_class(
    "param_change_batch_reduce",
    var_name_types=[("reducedChange", "scalar", VarAccess_REDUCE_BATCH_SUM)],
    var_refs=[("change", "scalar")],
    update_code="""
    $(reducedChange) = $(change);
    $(change) = 0;
    """)

update_param_change = create_custom_custom_update_class(
    "update_param_change",
    var_refs=[("change", "scalar"), ("variable", "scalar")],
    param_names=["batch_size", "low", "high"],
    update_code="""
    // Update
    $(variable) += $(change) / $(batch_size);
    $(variable) = min($(high), max($(low), $(variable)));
    """)

update_param_change_momentum = create_custom_custom_update_class(
    "update_weight_change_momentum",
    var_refs=[("change", "scalar"), ("variable", "scalar")],
    var_name_types=[("m", "scalar")],
    param_names=["batch_size", "beta", "low", "high"],
    update_code="""
    const scalar change_norm = $(change) / $(batch_size);
    $(m) = $(beta) * $(m) + (1.0-$(beta)) * change_norm;
    $(variable) += $(m);
    $(variable) = min($(high), max($(low), $(variable)));
    """
)

adam_optimizer_model = create_custom_custom_update_class(
    "adam_optimizer",
    param_names=["lr", "beta1", "beta2", "epsilon", "batch_size", "low", "high"],
    var_name_types=[("m", "scalar"), ("v", "scalar"), ("time", "scalar")],
    var_refs=[("change", "scalar"), ("variable", "scalar")],
    update_code="""
    // Update biased first moment estimate
    const scalar change_norm = $(change) / $(batch_size);
    $(m) = $(beta1) * $(m) + (1.0 - $(beta1)) * change_norm;
    // Update biased second moment estimate
    $(v) = $(beta2) * $(v) + (1.0 - $(beta2)) * change_norm * change_norm;
    // Add gradient to variable, scaled by learning rate
    const scalar m_hat = $(m) / (1.0 - pow($(beta1),$(time)));
    const scalar v_hat = $(v) / (1.0 - pow($(beta2),$(time)));
    $(variable) += $(lr) * m_hat / (sqrt(v_hat) + $(epsilon));
    $(variable) = min($(high), max($(low), $(variable)));
    $(time) += 1.0;
    """)

optimizers = {
    "sgd": {
        "model": update_param_change,
        "var_init": {}
    },
    "sgd_momentum": {
        "model": update_param_change_momentum,
        "var_init": {"m": 0.0}
    },
    "adam": {
        "model": adam_optimizer_model,
        "var_init": {"m": 0.0, "v": 0.0, "time": 1.0}
    }
}

def merge_dicts(dict_1, dict_2):

    return dict_1 | dict_2



def merge_dict_list_strings(dict_1, dict_2, key):
    '''
    
    '''
    # check if both lists provide lists associated
    # with the given key if the key is present
    assert isinstance(dict_1.get(key, []), list), \
        f"value of {key} in list 1 is not a list."
    assert isinstance(dict_2.get(key, []), list), \
        f"value of {key} in list 2 is not a list."
    
    # check if every element in the lists are
    # are strings.        
    assert all(isinstance(s, str) for s in dict_1.get(key, [])
               ), f"{key} list 1 contains non-string elements"
    assert all(isinstance(s, str) for s in dict_2.get(key, [])
               ), f"{key} list 2 contains non-string elements"

    return list(set(dict_1.get(key, [])+dict_2.get(key, [])))


def merge_dict_tuple_args(dict_1, dict_2, key):

    dict_list_1 = dict(dict_1.get(key, []))
    dict_list_2 = dict(dict_2.get(key, []))

    assert all(isinstance(s, str)
               for s in dict_list_1.keys()), "List 1 contains non-string names"
    assert all(isinstance(s, str)
               for s in dict_list_2.keys()), "List 2 contains non-string names"

    dict_merge = merge_dicts(dict_list_1, dict_list_2)

    return list(dict_merge.items())


def merge_wu_def(class_name, def_1, def_2):

    if def_1 is None:
        def_1 = {}
    if def_2 is None:
        def_2 = {}

    wu_def = {
        "class_name": class_name
    }

    wu_def["param_names"] = merge_dict_list_strings(
        def_1, def_2, "param_names")

    var_params_pair_names = ["var_name_types",
                             "pre_var_name_types",
                             "post_var_name_types",
                             "derived_params",
                             "extra_global_params"]

    for vp in var_params_pair_names:
        wu_def[vp] = merge_dict_tuple_args(def_1, def_2, vp)

    code_strings = ["sim_code",
                    "event_code",
                    "learn_post_code",
                    "synapse_dynamics_code",
                    "event_threshold_condition_code",
                    "pre_spike_code",
                    "post_spike_code",
                    "pre_dynamics_code",
                    "post_dynamics_code",
                    "sim_support_code",
                    "learn_post_support_code",
                    "synapse_dynamics_suppport_code"]

    for code in code_strings:
        code_1 = def_1.get(code, None)
        code_2 = def_2.get(code, None)

        # if both codes are None, set the resulting
        # code to None.
        if code_1 is None and code_2 is None:
            wu_def[code] = None
        else:
            # else, set either code variables to
            # an empty string if they are None...
            code_1 = "" if code_1 is None else code_1
            code_2 = "" if code_2 is None else code_2

            # ... and merge both string with a line break.
            wu_def[code] = f'''{code_1}\n{code_2}'''

    # check additional boolean requirements parameters
    boolean_flags = ["is_pre_spike_time_required",
                     "is_post_spike_time_required",
                     "is_pre_spike_event_time_required",
                     "is_prev_pre_spike_time_required",
                     "is_prev_post_spike_time_required",
                     "is_prev_pre_spike_event_time_required"]

    for flag in boolean_flags:
        wu_def[flag] = (def_1.get(flag, None) is True) or (
            def_2.get(flag, None) is True)

    return wu_def

def merge_ps_def(class_name, def_1, def_2):

    if def_1 is None:
        def_1 = {}
    if def_2 is None:
        def_2 = {}

    ps_def = {
        "class_name": class_name
    }

    ps_def["param_names"] = merge_dict_list_strings(
        def_1, def_2, "param_names")

    var_params_pair_names = ["var_name_types",
                             "derived_params",
                             "extra_global_params"]

    for vp in var_params_pair_names:
        ps_def[vp] = merge_dict_tuple_args(def_1, def_2, vp)

    code_strings = ["decay_code",
                    "apply_input_code",
                    "support_code"]

    for code in code_strings:
        code_1 = def_1.get(code, None)
        code_2 = def_2.get(code, None)

        # if both codes are None, set the resulting
        # code to None.
        if(code_1 is None and code_2 is None):
            ps_def[code] = None
        else:
            # else, set either code variables to
            # an empty string if they are None...
            code_1 = "" if code_1 is None else code_1
            code_2 = "" if code_2 is None else code_2

            # ... and merge both string with a line break.
            ps_def[code] = f'''{code_1}\n{code_2}'''

    return ps_def
