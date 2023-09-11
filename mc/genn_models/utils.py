import re
from copy import deepcopy

#######################
"""define the transmission part of
both the event-based model as well as
the continuous/rate model.
"""

WU_TRANSMIT = {
    "event": {
        "class_name": "weight_update_model_transmit_change",
        "param_names": [],
        "var_name_types": [("g", "scalar"), ("inp_prev", "scalar")],
        "sim_code": """
            const scalar inp = $(g)*$(r_pre);
            $(addToInSyn, inp-$(inp_prev));
            $(inp_prev) = inp;
        """
    },
    "cont": {
        "class_name": "weight_update_model_transmit_cont",
        "param_names": [],
        "var_name_types": [("g", "scalar")],
        "synapse_dynamics_code": """
            $(addToInSyn, $(g) * $(r_pre));
        """
    }

}

WU_VAR_SPACE_TRANSMIT = {
    "event": {"inp_prev": 0.0},
    "cont": {}
}

WU_PARAM_SPACE_TRANSMIT = {
    "event": {},
    "cont": {}
}

###########################
""" postsynaptic models matching
the weight update definitions.
"""

PS_TRANSMIT = {
    "event": {
        "class_name": "postsynaptic_change",
        "param_names": [],
        "var_name_types": [],
        "apply_input_code": """
            $(Isyn) += $(inSyn);
        """
    },
    "cont": {
        "class_name": "postsynaptic_cont",
        "param_names": [],
        "var_name_types": [],
        "apply_input_code": "$(Isyn) += $(inSyn); $(inSyn) = 0.0;"
    }
}

PS_VAR_SPACE_TRANSMIT = {
    "event": {},
    "cont": {}
}

PS_PARAM_SPACE_TRANSMIT = {
    "event": {},
    "cont": {}
}

WU_PARAM_SPACE_PLAST = {
    "event": {},
    "cont": {}
}

###################
""" default values for
**mandatory** plasticity
variables dg (changes in weights g)
and dg_prev (storing the previous
value of dg).
"""

WU_VAR_SPACE_PLAST = {
    "event": {
        "dg": 0.0,
        "dg_prev": 0.0
    },
    "cont": {
        "dg": 0.0
    }
}


def find_post_vars(s):
    """find all postsynaptic variables
    used in a string, using a regexp:
    matches substrings of the form
    $(<variable name>_post).
    """
    post_vars = []
    regstr = "\$\([^\)]*_post\)"
    while True:
        srch = re.search(regstr, s)
        if bool(srch):
            _vr = srch.group()
            post_vars.append(_vr[2:-6])
            s = "".join(s.split(_vr))
        else:
            break
    return post_vars


def convert_f_prev_step(s):
    """convert a string defining the continuous
    weight update rule into the event-based approach.
    This involves replacing $(r_pre) with $(r_prev_event_pre)
    and all postsynaptic variables with their
    $(<variable name>_prev_post) counterpart (which has to
    be preseint in the postsynaptic neuron model, of course).
    """
    # replace $(r_pre) if present
    s = "$(r_prev_event_pre)".join(s.split("$(r_pre)"))

    post_vars = find_post_vars(s)

    for _var in post_vars:
        _var_full_old = f"$({_var}_post)"
        _var_full_new = f"$({_var}_prev_event_post)"
        s = _var_full_new.join(s.split(_var_full_old))
    
    import ipdb
    ipdb.set_trace()
    return s


def generate_event_plast_wu_dict(name, f, params=[], extra_vars=[]):
    """generate the event-based version of
    the plasticity weight update definition"""

    var_name_types = [("dg", "scalar"), ("dg_prev", "scalar")] + extra_vars

    if f in (None, ""):
        sim_code = f"//PLAST CODE {name} \n"
        spk_requ = False
    else:
        #f_prev = convert_f_prev_step(f)

        sim_code = f"""
            // PLAST CODE {name}

            $(dg) += ($(t) - max(0.0,$(prev_sT_pre))) * $(dg_prev);

            $(dg_prev) = {f};
        """
        spk_requ = True

    w_update_model_plast = {
        "class_name": name,
        "param_names": params,
        "var_name_types": var_name_types,
        "sim_code": sim_code,
        "is_prev_pre_spike_time_required": spk_requ,
    }

    return w_update_model_plast


def generate_cont_plast_wu_dict(name, f, params=[], extra_vars=[]):
    """generate the continuous version of
    the plasticity weight update definition"""

    var_name_types = [("dg", "scalar")] + extra_vars

    if f in (None, ""):
        synapse_dynamics_code = f"//PLAST CODE {name} \n"
    else:

        synapse_dynamics_code = f"""
            // PLAST CODE {name}

            const scalar dg_temp = {f};

            // previous version:
            $(dg) += DT * (dg_temp);
        """

    w_update_model_plast = {
        "class_name": name,
        "param_names": params,
        "var_name_types": var_name_types,
        "synapse_dynamics_code": synapse_dynamics_code
    }

    return w_update_model_plast


def generate_plast_wu_dict(mod_type, name, f, params=[], extra_vars=[]):
    """wrapper function returning the respective plasticity
    weight update definition depending on the model type"""

    assert mod_type in ["event", "cont"], \
        "given type must be 'event' or 'cont'"

    if mod_type == "event":
        return generate_event_plast_wu_dict(name, f, params, extra_vars)
    else:
        return generate_cont_plast_wu_dict(name, f, params, extra_vars)

###################################


def convert_event_neur_dict(neur_model_dict, post_plast_vars):
    """convert a "continous" neuron model definition into
    a variant suitable for event-based synapses. This involves
    adding a $(<variable name>_prev) for every variable listed
    in post_plast_vars, i.e. postsynaptic variables accessed
    in the weight update. Moreover, the variables r_event and
    r_prev_event are added as they are required for event-based
    synaptic transmission. Finally, the threshold condition and
    reset code is added.
    """

    neur_model_dict = deepcopy(neur_model_dict)

    #event_vars = list(set(post_plast_vars + ["r"]))

    neur_model_dict["var_name_types"] = (neur_model_dict["var_name_types"]
            + [("r_event", "scalar")])

    #neur_model_dict["var_name_types"] = (neur_model_dict["var_name_types"]
    #    + [(f"{var}_prev_event", "scalar") for var in event_vars]
    #    + [(f"{var}_event", "scalar") for var in event_vars])

    neur_model_dict["param_names"] = neur_model_dict["param_names"] + ["th"]

    neur_model_dict["threshold_condition_code"] = "abs($(r) - $(r_event)) >= $(th)"

    neur_model_dict["reset_code"] = "$(r_event) = $(r);"

    #neur_model_dict["reset_code"] = "\n".join([
    #f"""
    #    $({var}_event) = $({var})""" for var in event_vars])

    sim_code = neur_model_dict["sim_code"]

    #sim_code_prev_update = "\n".join([
    #f"""
    #$({var}_prev_event) = $({var}_event);""" for var in event_vars])

    #neur_model_dict["sim_code"] = f"{sim_code_prev_update} \n {sim_code}"
    #neur_model_dict["sim_code"] = sim_code

    neur_model_dict["is_auto_refractory_required"] = False

    return neur_model_dict


def convert_event_neur_var_space_dict(var_space, post_plast_vars):
    """convert the variable initialization of a continuous
    model into the event-based variant."""

    var_space = deepcopy(var_space) | {"r_event": var_space["r"]}

    #event_vars = list(set(post_plast_vars + ["r"]))

    #var_space = dict(var_space) | dict([(f"{var}_event", var_space[var]) for var in event_vars]
    #                                + [(f"{var}_event", var_space[var]) for var in event_vars])
    return var_space


def convert_event_neur_param_space_dict(param_space, th=1e-3):
    param_space = dict(param_space) | {"th": th}

    return param_space


def convert_neuron_mod_data_cont_to_event(mod_dat, post_plast_vars, th=1e-3):
    """wrapper function that returns the converted model definition,
    variable, and parameter space."""

    mod_dat = dict(mod_dat)

    mod_dat["model_def"] = convert_event_neur_dict(mod_dat["model_def"],
                                                   post_plast_vars)

    mod_dat["param_space"] = convert_event_neur_param_space_dict(mod_dat["param_space"], th)

    mod_dat["var_space"] = convert_event_neur_var_space_dict(mod_dat["var_space"], post_plast_vars)

    return mod_dat
