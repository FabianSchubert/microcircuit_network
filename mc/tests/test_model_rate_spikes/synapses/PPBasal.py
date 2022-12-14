from ..utils import (WU_TRANSMIT, WU_TRANSMIT_VAR, WU_TRANSMIT_PARAM,
                     PS_TRANSMIT, PS_TRANSMIT_VAR, PS_TRANSMIT_PARAM)
from pygenn.genn_model import init_var

w_update_model_transmit = dict(WU_TRANSMIT)
wu_param_space_transmit = dict(WU_TRANSMIT_PARAM)
wu_var_space_transmit = dict(WU_TRANSMIT_VAR)
wu_var_space_transmit["g"] = init_var("Uniform", {"min": 0.0, "max": 1.0})

w_update_model_plast = {
    "class_name": "weight_update_model_pyr_to_pyr_fwd_def",
    "param_names": ["muPP_basal"],
    "var_name_types": [("g", "scalar"), ("vbEff", "scalar")],
    "sim_code": f"""
        // SIM CODE PP BASAL
        $(vbEff) = $(vb_post) * $(gb_post)/($(glk_post)+$(gb_post)+$(ga_post));
        $(g) += $(muPP_basal)*($(u_post) - $(vbEff));
    """
}

wu_param_space_plast = {"muPP_basal": 0}#4e-3}

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