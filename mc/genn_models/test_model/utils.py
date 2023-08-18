"""
Some helper functions for creating the model definitions.
"""

TH_COND_CODE = "$(gennrand_uniform) <= $(r)*DT"

RESET_CODE = ""

WU_TRANSMIT = {
    "class_name": "weight_update_model_transmit_spike",
    "var_name_types": [("g", "scalar")],
    "sim_code": """
        $(addToInSyn, $(g));
    """
}

WU_TRANSMIT_VAR = {
    # intentionally left blank to enforce manual setting for each synapse type
}

WU_TRANSMIT_PARAM = {}

PS_TRANSMIT = {
    "class_name": "integrator_update",
    "param_names": [],
    "apply_input_code": "$(Isyn) += $(inSyn); $(inSyn) =  0.0;"
}

PS_TRANSMIT_VAR = {}

PS_TRANSMIT_PARAM = {}


def act_func(x):
    return f'max(0.0,{x})'

