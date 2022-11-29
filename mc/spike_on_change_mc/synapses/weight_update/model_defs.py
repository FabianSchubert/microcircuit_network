'''
Weight update model definitions stored in
dicts.
'''

from ...utils import act_func

wu_model_transmit_rate = {
    "class_name": "weight_update_model_transmit_rate",
    "param_names": [],
    "var_name_types": [("g", "scalar")],
    "synapse_dynamics_code": "$(addToInSyn, $(g) * $(r_pre));",
}

wu_model_transmit_rate_diff = {
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

wu_model_pp_basal = {
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
    # '''
    # "synapse_dynamics_code": f"""
    #    $(vbEff) = $(vb_post) * $(gb_post)/($(glk_post)+$(gb_post)+$(ga_post));
    #    $(dg) += DT * ($(muPP_basal) * $(r_last_pre)
    #    * ($(r_last_post) - {act_func('$(vbEff)')} ) - $(dg))/$(tau);
    #    $(g) += DT * $(dg);
    # """,'''
    "is_prev_pre_spike_time_required": True,
    "is_prev_post_spike_time_required": True,
    "is_pre_spike_time_required": True,
    "is_post_spike_time_required": True
}

wu_model_pp_apical = {
    "class_name": "weight_update_model_pyr_to_pyr_back_def",
    "param_names": ["muPP_apical"],
    "var_name_types": [("g", "scalar")],
    "synapse_dynamics_code": "",
    "is_pre_spike_time_required": False,
    "is_post_spike_time_required": False
}

wu_model_pinp = {
    "class_name": "weight_update_model_input_to_pyr",
    "param_names": ["muPINP", "tau"],
    "var_name_types": [("g", "scalar"), ("dg", "scalar"), ("vbEff", "scalar"), ("t_last", "scalar")],
    "sim_code": f"""
        // SIM CODE PINP
        $(vbEff) = $(vb_post) * $(gb_post)/($(glk_post)+$(gb_post)+$(ga_post));
        const scalar dt = $(t) - $(t_last);
        if(dt > 0){{
            const scalar dw = $(r_prev_last_pre)*($(r_prev_last_post) - {act_func('$(vbEff)')});
            const scalar dg_prev = $(dg);
            $(dg) = dw + (dg_prev - dw) * exp(-dt/$(tau));
            $(g) += $(muPINP) * (dt * dw - $(tau) * ($(dg) - dg_prev));
            $(t_last) = $(t);
        }}
    """,
    "learn_post_code": f"""
        // LEARN POST CODE PINP
        $(vbEff) = $(vb_post) * $(gb_post)/($(glk_post)+$(gb_post)+$(ga_post));
        const scalar dt = $(t) - $(t_last);
        if(dt > 0){{
            const scalar dw = $(r_prev_last_pre)*($(r_prev_last_post) - {act_func('$(vbEff)')});
            const scalar dg_prev = $(dg);
            $(dg) = dw + (dg_prev - dw) * exp(-dt/$(tau));
            $(g) += $(muPINP) * (dt * dw - $(tau) * ($(dg) - dg_prev));
            $(t_last) = $(t);
        }}
    """,
    # '''
    # "synapse_dynamics_code": f"""
    #     $(vbEff) = $(vb_post) * $(gb_post)/($(glk_post)+$(gb_post)+$(ga_post));
    #     $(dg) += DT * ($(muPINP) * $(r_last_pre)
    #     * ($(r_last_post) - {act_func('$(vbEff)')} ) - $(dg))/$(tau);
    #     $(g) += DT * $(dg);
    # """,'''
    "is_prev_pre_spike_time_required": True,
    "is_prev_post_spike_time_required": True,
    "is_pre_spike_time_required": True,
    "is_post_spike_time_required": True
}

wu_model_ip = {
    "class_name": "weight_update_model_pyr_to_int_def",
    "param_names": ["muIP", "tau"],
    "var_name_types": [("g", "scalar"), ("dg", "scalar"), ("vEff", "scalar"), ("t_last", "scalar")],
    "sim_code": f"""
        // SIM CODE IP
        $(vEff) = $(v_post) * $(gd_post)/($(glk_post)+$(gd_post));
        const scalar dt = $(t) - $(t_last);
        if(dt > 0){{
            const scalar dw = $(r_prev_last_pre)*($(r_prev_last_post) - {act_func('$(vEff)')});
            const scalar dg_prev = $(dg);
            $(dg) = dw + (dg_prev - dw) * exp(-dt/$(tau));
            $(g) += $(muIP) * (dt * dw - $(tau) * ($(dg) - dg_prev));
            $(t_last) = $(t);
        }}
    """,
    "learn_post_code": f"""
        // LEARN POST CODE IP
        $(vEff) = $(v_post) * $(gd_post)/($(glk_post)+$(gd_post));
        const scalar dt = $(t) - $(t_last);
        if(dt > 0){{
            const scalar dw = $(r_prev_last_pre)*($(r_prev_last_post) - {act_func('$(vEff)')});
            const scalar dg_prev = $(dg);
            $(dg) = dw + (dg_prev - dw) * exp(-dt/$(tau));
            $(g) += $(muIP) * (dt * dw - $(tau) * ($(dg) - dg_prev));
            $(t_last) = $(t);
        }}
    """,
    #'''
    #"synapse_dynamics_code": f"""
    #    $(vEff) = $(v_post) * $(gd_post)/($(glk_post)+$(gd_post));
    #    $(dg) += DT * ($(muIP) * $(r_last_pre)
    #        * ($(r_last_post) - {act_func('$(vEff)')}) - $(dg))/$(tau);
#
#        $(g) += DT * $(dg);
#    """,'''
    "is_prev_pre_spike_time_required": True,
    "is_prev_post_spike_time_required": True,
    "is_pre_spike_time_required": True,
    "is_post_spike_time_required": True
}

wu_model_ip_back = {
    "class_name": "weight_update_model_pyr_to_int_back_def",
    "param_names": [],
    "var_name_types": [("g", "scalar")],
    "synapse_dynamics_code": "",
    "is_pre_spike_time_required": False,
    "is_post_spike_time_required": False
}

wu_model_pi = {
    "class_name": "weight_update_model_int_to_pyr_def",
    "param_names": ["muPI", "tau"],
    "var_name_types": [("g", "scalar"), ("dg", "scalar"), ("t_last", "scalar")],
    "sim_code": f"""
        // SIM CODE PI
        const scalar dt = $(t) - $(t_last);
        if(dt > 0){{
            const scalar dw = -$(r_prev_last_pre)*$(va_post);
            const scalar dg_prev = $(dg);
            $(dg) = dw + (dg_prev - dw) * exp(-dt/$(tau));
            $(g) += $(muPI) * (dt * dw - $(tau) * ($(dg) - dg_prev));
            $(t_last) = $(t);
        }}
    """,
    "learn_post_code": f"""
        // LEARN POST CODE PI
        const scalar dt = $(t) - $(t_last);
        if(dt > 0){{
            const scalar dw = -$(r_prev_last_pre)*$(va_post);
            const scalar dg_prev = $(dg);
            $(dg) = dw + (dg_prev - dw) * exp(-dt/$(tau));
            $(g) += $(muPI) * (dt * dw - $(tau) * ($(dg) - dg_prev));
            $(t_last) = $(t);
        }}
    """,
    #'''
    #"synapse_dynamics_code": """
    #    $(dg) += DT * (-$(muPI) * $(r_last_pre)
    #        * $(va_post) - $(dg));
    #    $(g) += DT * $(dg);
    #""",'''
    "is_prev_pre_spike_time_required": True,
    "is_prev_post_spike_time_required": True,
    "is_pre_spike_time_required": True,
    "is_post_spike_time_required": True
}
