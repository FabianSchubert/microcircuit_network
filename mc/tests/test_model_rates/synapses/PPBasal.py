from ..utils import act_func
from pygenn.genn_model import init_var

w_update_model_transmit = {
    "class_name": "weight_update_model_transmit_rate",
    "param_names": [],
    "var_name_types": [("g", "scalar")],
    "synapse_dynamics_code": "$(addToInSyn, $(g) * $(r_pre));"
}

w_update_model_plast = {
    "class_name": "weight_update_model_pyr_to_pyr_fwd_def",
    "param_names": ["muPP_basal", "tau"],
    "var_name_types": [("g", "scalar"), ("dg", "scalar"), ("vbEff", "scalar")],
    "synapse_dynamics_code": f"""
        // SIM CODE PP BASAL
        $(vbEff) = $(vb_post) * $(gb_post)/($(glk_post)+$(gb_post)+$(ga_post));
        
        $(dg) += DT * ($(muPP_basal) * $(r_pre) * ($(r_post) - {act_func('$(vbEff)')}) - $(dg)) / $(tau);
        //$(g) += DT * $(muPP_basal) * $(dg);
    """,
    "is_prev_pre_spike_time_required": False,
    "is_prev_post_spike_time_required": False,
    "is_pre_spike_time_required": False,
    "is_post_spike_time_required": False
}

wu_param_space_transmit = {}

wu_param_space_plast = {"muPP_basal": 8*5e-4,
                           "tau": 30.}

wu_var_space_transmit = {
    "g": init_var("Uniform", {"min": -1.0, "max": 1.0})
}

wu_var_space_plast = {
    "vbEff": 0.0,
    "dg": 0.0
}

ps_model_transmit = {
    "class_name": "delta_update",
    "apply_input_code": "$(Isyn) += $(inSyn); $(inSyn) = 0.0;"
}

ps_param_space_transmit = {}

ps_var_space_transmit = {}

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
    "norm_after_init": "sqrt"
}