"""
Some helper functions for creating the model definitions.
"""

TH_COND_CODE = """
    abs($(r) - $(r_prev)) >= $(th)
"""

RESET_CODE = """
    $(r_prev_prev) = $(r_prev);
    $(r_prev) = $(r);
    
    $(r_eff_prev_prev) = $(r_eff_prev);
    $(r_eff_prev) = $(r_eff);
"""

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

WU_TRANSMIT_VAR = {
    # intentionally left blank to enforce manual setting for each synapse type
}

WU_TRANSMIT_PARAM = {}

PS_TRANSMIT = {
    "class_name": "postsynaptic_change",
    "param_names": [],
    "var_name_types": [],
    "apply_input_code": """
        $(Isyn) += $(inSyn);
    """
}

PS_TRANSMIT_VAR = {}

PS_TRANSMIT_PARAM = {}

def act_func(x):
    return f'max(0.0, {x})'