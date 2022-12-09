from ..utils import act_func
from pygenn.genn_model import init_var

w_update_model_transmit = {
    "class_name": "weight_update_model_transmit_spike",
    "param_names": [],
    "var_name_types": [("g", "scalar"),
                       ("w_input_last", "scalar"),
                       ("w_input", "scalar")],
    "sim_code": """
        // calculate new weighted input
        $(w_input) = $(g) * $(r_pre);
        // This one should add the difference to inSyn
        $(addToInSyn, $(w_input) - $(w_input_last));
        // This one should update w_input_last
        $(w_input_last) = $(w_input);
    """
}

w_update_model_plast = {
    "class_name": "weight_update_model_pyr_to_pyr_fwd_def",
    "param_names": ["muPP_basal", "tau"],
    "var_name_types": [("g", "scalar"), ("dg", "scalar"), ("vbEff", "scalar"), ("t_last", "scalar")],
    "sim_code": f"""
        // SIM CODE PP BASAL
        $(vbEff) = $(vb_post) * $(gb_post)/($(glk_post)+$(gb_post)+$(ga_post));
        const scalar dt = $(t) - $(t_last);
        if(dt > 0){{
            const scalar dw = $(r_prev_last_pre)*($(r_prev_last_post) - {act_func('$(vbEff)')});
            const scalar dg_prev = $(dg);
            $(dg) = dw + (dg_prev - dw) * exp(-dt/$(tau));
            $(g) += $(muPP_basal) * (dt * dw - $(tau) * ($(dg) - dg_prev));
            $(t_last) = $(t);
        }}
        
    """,
    "learn_post_code": f"""
        // LEARN POST CODE PP BASAL
        $(vbEff) = $(vb_post) * $(gb_post)/($(glk_post)+$(gb_post)+$(ga_post));
        const scalar dt = $(t) - $(t_last);
        if(dt > 0){{
            const scalar dw = $(r_prev_last_pre)*($(r_prev_last_post) - {act_func('$(vbEff)')});
            const scalar dg_prev = $(dg);
            $(dg) = dw + (dg_prev - dw) * exp(-dt/$(tau));
            $(g) += $(muPP_basal) * (dt * dw - $(tau) * ($(dg) - dg_prev));
            $(t_last) = $(t);
        }}
    """,
    "is_prev_pre_spike_time_required": True,
    "is_prev_post_spike_time_required": True,
    "is_pre_spike_time_required": True,
    "is_post_spike_time_required": True
}

wu_param_space_transmit = {}

wu_param_space_plast = {"muPP_basal": 4e-3,
                           "tau": 5.}

wu_var_space_transmit = {
    "g": init_var("Uniform", {"min": -1.0, "max": 1.0}),
    "w_input": 0.0,
    "w_input_last": 0.0
}

wu_var_space_plast = {
    "g": init_var("Uniform", {"min": -1.0, "max": 1.0}),
    "vbEff": 0.0,
    "dg": 0.0,
    "t_last": 0.0
}

ps_model_transmit = {
    "class_name": "integrator_update",
    "apply_input_code": "$(Isyn) += $(inSyn);"
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
    "ps_var_space_transmit": ps_var_space_transmit
}