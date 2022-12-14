"""
Some helper functions for creating the model definitions.
"""
from pygenn.genn_model import init_var

#TH_COND_CODE = "$(u) >= $(u_th)"
#TH_COND_CODE = "$(t)-$(t_last_spike) >= 1./$(r)"
TH_COND_CODE = "$(gennrand_uniform) <= $(r)*DT"

#RESET_CODE = "$(u) = $(u_r);"
#RESET_CODE = "$(t_last_spike) = $(t);"
RESET_CODE = ""

TAU_TRANSMIT = 1.0

WU_TRANSMIT = {
    "class_name": "weight_update_model_transmit_spike",
    "param_names": ["tau"],
    "var_name_types": [("g", "scalar")],
    "sim_code": """
        $(addToInSyn, $(g)/$(tau));
    """
}

WU_TRANSMIT_VAR = {
    # intentionally left blank to enforce manual setting for each synapse type
}

WU_TRANSMIT_PARAM = {"tau": TAU_TRANSMIT}

PS_TRANSMIT = {
    "class_name": "integrator_update",
    "param_names": ["tau"],
    "apply_input_code": "$(Isyn) += $(inSyn); $(inSyn) = $(inSyn)*(1.-DT/$(tau));"
}

PS_TRANSMIT_VAR = {}

PS_TRANSMIT_PARAM = {"tau": TAU_TRANSMIT}


def act_func(x):
    #return f'3.0*log(1.0+exp(1.0*({x})))'
    return f'5.0*max(0.0,{x})'

