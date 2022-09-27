#! /usr/bin/env python3

from ..utils import act_func

wu_model_pp_basal = {
	"class_name": "weight_update_model_exc_to_exc_fwd_def",
	"param_names": ["muPP"],
	"var_name_types": [("g", "scalar"),("vbEff","scalar")],
	"synapse_dynamics_code": f"""
		$(addToInSyn, $(g) * $(r_pre));
		$(vbEff) = $(vb_post) * $(gb_post)/($(glk_post)+$(gb_post)+$(ga_post));
		$(g) += DT * $(muPP) * $(r_pre)
		* ($(r_post) - {act_func('$(vbEff)')} );
	""",
	"is_pre_spike_time_required": False,
	"is_post_spike_time_required": False
}

wu_model_pp_apical = {
	"class_name": "weight_update_model_exc_to_exc_back_def",
	"param_names": ["muPP"],
	"var_name_types": [("g", "scalar")],
	"synapse_dynamics_code": f"""
	    $(addToInSyn, $(g) * $(r_pre));
	    $(g) += DT * $(muPP) * $(r_pre)
	        * ($(r_post) - $(va_exc_post));
	""",
	"is_pre_spike_time_required": False,
	"is_post_spike_time_required": False
}

wu_model_pi = {
    "class_name": "weight_update_model_exc_to_int_def",
    "param_names": ["muIP"],
    "var_name_types": [("g", "scalar"),("vEff","scalar")],
    "synapse_dynamics_code": f"""
        $(addToInSyn, $(g) * $(r_pre));
        $(vEff) = $(v_post) * $(gd_post)/($(glk_post)+$(gd_post));
        $(g) += DT * $(muIP) * $(r_pre)
            * ($(r_post) - {act_func('$(vEff)')});
    """,
    "is_pre_spike_time_required": False,
    "is_post_spike_time_required": False
}

wu_model_ip = {
    "class_name": "weight_update_model_int_to_exc_def",
    "param_names": ["muPI"],
    "var_name_types": [("g", "scalar")],
    "synapse_dynamics_code": """
        $(addToInSyn, $(g) * $(r_pre));
        $(g) -= DT * $(muPI) * $(r_pre)
            * $(va_post);
    """,
    "is_pre_spike_time_required": False,
    "is_post_spike_time_required": False
}