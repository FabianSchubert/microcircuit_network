from ..utils import (WU_TRANSMIT, WU_TRANSMIT_VAR, WU_TRANSMIT_PARAM,
                     PS_TRANSMIT, PS_TRANSMIT_VAR, PS_TRANSMIT_PARAM)

from pygenn.genn_model import init_var

# reinit a dict instance, otherwise there is
# a risk of overwriting other stuff
w_update_model_transmit = dict(WU_TRANSMIT)
wu_param_space_transmit = dict(WU_TRANSMIT_PARAM)
wu_var_space_transmit = dict(WU_TRANSMIT_VAR)
WEIGHT_SCALE = 0.5
wu_var_space_transmit["g"] = init_var("Uniform", {"min": -WEIGHT_SCALE, "max": WEIGHT_SCALE})
wu_var_space_transmit["inp_prev"] = 0.0

w_update_model_plast = {
    "class_name": "weight_update_model_pyr_to_int",
    "param_names": ["muIP"],
    "var_name_types": [("g", "scalar"), ("dg", "scalar")],
    "sim_code": f"""
        // SIM CODE IP
        $(dg) += $(muIP) * ($(t) - max(max(0.0,$(prev_sT_pre)),$(sT_post))) * $(r_prev_prev_pre) * ($(r_prev_post) - $(r_eff_prev_post));
    """,
    "learn_post_code": f"""
        // LEARN POST CODE IP
        $(dg) += $(muIP) * ($(t) - max(max(0.0,$(prev_sT_post)),$(sT_pre))) * $(r_prev_pre) * ($(r_prev_prev_post) - $(r_eff_prev_prev_post));
    """,
    "is_pre_spike_time_required": True,
    "is_post_spike_time_required": True,
    "is_prev_pre_spike_time_required": True,
    "is_prev_post_spike_time_required": True
}

wu_param_space_plast = {"muIP": 0e-3}

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
    "norm_after_init": "sqrt"
}