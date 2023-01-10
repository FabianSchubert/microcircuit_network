from pygenn.genn_model import init_var
from ..utils import (WU_TRANSMIT, WU_TRANSMIT_VAR, WU_TRANSMIT_PARAM,
                     PS_TRANSMIT, PS_TRANSMIT_VAR, PS_TRANSMIT_PARAM)

w_update_model_transmit = dict(WU_TRANSMIT)
wu_param_space_transmit = dict(WU_TRANSMIT_PARAM)
wu_var_space_transmit = dict(WU_TRANSMIT_VAR)
WEIGHT_SCALE = 0.25
wu_var_space_transmit["g"] = init_var("Uniform", {"min": -WEIGHT_SCALE, "max": 0.0})

w_update_model_plast = {
    "class_name": "weight_update_model_int_to_pyr_def",
    "param_names": ["muPI"],
    "var_name_types": [("g", "scalar")],
    "sim_code": f"""
        // SIM CODE PI
        $(g) += -$(muPI) * $(va_post);
        $(g) = min(0.,$(g));
    """
}

wu_param_space_plast = {"muPI": 1*0.25*8e-4}

wu_var_space_plast = {
    #"g": init_var("Uniform", {"min": -1.0, "max": 0.0})
}

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
    "norm_after_init": "lin"
}