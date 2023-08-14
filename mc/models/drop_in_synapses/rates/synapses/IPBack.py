from ..utils import (WU_TRANSMIT, WU_TRANSMIT_VAR, WU_TRANSMIT_PARAM,
                     PS_TRANSMIT, PS_TRANSMIT_VAR, PS_TRANSMIT_PARAM)

w_update_model_transmit = dict(WU_TRANSMIT)
wu_param_space_transmit = dict(WU_TRANSMIT_PARAM)
wu_var_space_transmit = dict(WU_TRANSMIT_VAR)
wu_var_space_transmit["g"] = 1.0

w_update_model_plast = {
    "class_name": "weight_update_model_pyr_to_int_back",
    "param_names": [],
    "var_name_types": [("g", "scalar"), ("dg", "scalar")],
    "is_pre_spike_time_required": False,
    "is_post_spike_time_required": False
}

wu_param_space_plast = {}

wu_var_space_plast = {"dg": 0.0}

ps_model_transmit = dict(PS_TRANSMIT)
ps_param_space_transmit = dict(PS_TRANSMIT_PARAM)
ps_var_space_transmit = dict(PS_TRANSMIT_VAR)

mod_dat = {
    "w_update_model_transmit": w_update_model_transmit,
    "w_update_model_plast": w_update_model_plast,
    "wu_param_space_transmit": wu_param_space_transmit,
    "wu_param_space_plast": wu_param_space_plast,
    "wu_var_space_transmit": wu_var_space_transmit,
    "wu_var_space_plast": wu_var_space_plast,
    "ps_model_transmit": ps_model_transmit,
    "ps_param_space_transmit": ps_param_space_transmit,
    "ps_var_space_transmit": ps_var_space_transmit,
    "norm_after_init": False
}