from ..utils import (WU_TRANSMIT, WU_TRANSMIT_VAR, WU_TRANSMIT_PARAM,
                     PS_TRANSMIT, PS_TRANSMIT_VAR, PS_TRANSMIT_PARAM)
from pygenn.genn_model import init_var

w_update_model_transmit = dict(WU_TRANSMIT)
wu_param_space_transmit = dict(WU_TRANSMIT_PARAM)
wu_var_space_transmit = dict(WU_TRANSMIT_VAR)
WEIGHT_SCALE = 0.5
wu_var_space_transmit["g"] = init_var("Uniform", {"min": 0.0, "max": WEIGHT_SCALE})

w_update_model_plast = {
    "class_name": "weight_update_model_pyr_to_pyr_fwd_def",
    "param_names": ["muPP_basal", "tau"],
    "var_name_types": [("g", "scalar"), ("dg", "scalar")],
    "sim_code": f"""
        // SIM CODE PP BASAL
        $(dg) += $(muPP_basal)*($(u_post) - $(vbEff_post)) / $(tau);
        //$(g) += $(muPP_basal)*($(u_post) - $(vbEff_post));
        //$(g) = max(0.,$(g));
    """,
    "synapse_dynamics_code": f"""
        $(dg) += -DT * $(dg) / $(tau);
    """
}

wu_param_space_plast = {"muPP_basal": 10*0.25*4e-2,
                        "tau": 20.0}

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
    "norm_after_init": "lin",
    "low": 0.0
}