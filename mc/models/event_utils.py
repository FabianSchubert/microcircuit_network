import re

WU_TRANSMIT = {
    "class_name": "weight_update_model_transmit_change",
    "param_names": [],
    "var_name_types": [("g", "scalar"), ("inp_prev", "scalar")],
    "sim_code": """
        const scalar inp = $(g)*$(r_pre);
        $(addToInSyn, inp-$(inp_prev));
        $(inp_prev) = inp;
    """
}

WU_VAR_SPACE_TRANSMIT = {
    "inp_prev": 0.0
}

WU_PARAM_SPACE_TRANSMIT = {}

PS_TRANSMIT = {
    "class_name": "postsynaptic_change",
    "param_names": [],
    "var_name_types": [],
    "apply_input_code": """
        $(Isyn) += $(inSyn);
    """
}

PS_VAR_SPACE_TRANSMIT = {}

PS_PARAM_SPACE_TRANSMIT = {}

####################################

def find_post_vars(s):
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
    ## replace $(r_pre) if present
    s = "$(r_prev_event_pre)".join(s.split("$(r_pre)"))

    post_vars = find_post_vars(s)

    for _var in post_vars:
        _var_full_old = f"$({_var}_post)"
        _var_full_new = f"$({_var}_prev_post)"
        s = _var_full_new.join(s.split(_var_full_old))

    return s

def generate_event_plast_wu_dict(name, f, params=[], extra_vars=[]):

    var_name_types = [("dg", "scalar"), ("dg_prev", "scalar")] + extra_vars

    if f in (None, ""):
        sim_code = f"//PLAST CODE {name} \n"
        spk_requ = False
    else:
        f_prev = convert_f_prev_step(f)

        sim_code = f"""
            // PLAST CODE {name}

            const scalar dg_temp = {f_prev};

            $(dg) += ($(t) - max(0.0,$(prev_sT_pre))) * ($(dg_prev) + dg_temp) * 0.5;

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

WU_PARAM_SPACE_PLAST = {}

WU_VAR_SPACE_PLAST = {
    "dg": 0.0,
    "dg_prev": 0.0
}

###################################

def convert_event_neur_dict(neur_model_dict, post_plast_vars):
    neur_model_dict = dict(neur_model_dict)

    neur_model_dict["var_name_types"] = (neur_model_dict["var_name_types"]
        + [(f"{var}_prev", "scalar") for var in post_plast_vars]
        + [("r_event", "scalar"), ("r_prev_event", "scalar")]
    )

    neur_model_dict["param_names"] = neur_model_dict["param_names"] + ["th"]

    neur_model_dict["threshold_condition_code"] = "abs($(r) - $(r_event)) >= $(th)"

    neur_model_dict["reset_code"] = """
    $(r_prev_event) = $(r_event);
    $(r_event) = $(r);
    """

    sim_code = neur_model_dict["sim_code"]

    sim_code_prev_update = "\n ".join([f"$({var}_prev) = $({var});" for var in post_plast_vars])

    neur_model_dict["sim_code"] = f"{sim_code_prev_update} \n {sim_code}"

    neur_model_dict["is_auto_refractory_required"] = False

    return neur_model_dict

def convert_event_neur_var_space_dict(var_space, post_plast_vars):
    var_space = dict(var_space) | dict([(f"{var}_prev", var_space[var]) for var in post_plast_vars]
                                    + [("r_event", 0.0), ("r_prev_event", 0.0)])
    return var_space

def convert_event_neur_param_space_dict(param_space):
    param_space = dict(param_space) | {"th": 1e-3}

    return param_space
