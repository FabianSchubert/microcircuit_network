
from ..utils import (WU_TRANSMIT, WU_TRANSMIT_VAR, WU_TRANSMIT_PARAM,
                     PS_TRANSMIT, PS_TRANSMIT_VAR, PS_TRANSMIT_PARAM)
from pygenn.genn_model import init_var

w_update_model_transmit = dict(WU_TRANSMIT)
wu_param_space_transmit = dict(WU_TRANSMIT_PARAM)
wu_var_space_transmit = dict(WU_TRANSMIT_VAR)
WEIGHT_SCALE = 0.25
wu_var_space_transmit["g"] = init_var("Uniform", {"min": 0.0, "max": WEIGHT_SCALE})

w_update_model_plast = {
    "class_name": "weight_update_model_input_to_pyr",
    "param_names": ["muPINP"],
    "var_name_types": [("g", "scalar"), ("vbEff", "scalar")],
    "sim_code": f"""
        // SIM CODE PINP
        //$(vbEff) = $(vb_post) * $(gb_post)/($(glk_post)+$(gb_post)+$(ga_post));
        $(g) += $(muPINP)*($(u_post) - $(vbEff_post));
        $(g) = max(0.,$(g));
    """
}



wu_param_space_plast = {"muPINP": 1*0.25*4e-3}

wu_var_space_plast = {
    #"g": init_var("Uniform", {"min": 0.0, "max": 1.0}),
    "vbEff": 0.0
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