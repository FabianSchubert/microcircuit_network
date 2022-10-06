#! /usr/bin/env python3

from ...utils import act_func

wu_model_pp_basal = {
	"class_name": "weight_update_model_pyr_to_pyr_fwd_def",
	"param_names": ["muPP_basal","tau"],
	"var_name_types": [("g", "scalar"),("dg","scalar"),("vbEff","scalar")],
	"synapse_dynamics_code": f"""
		$(addToInSyn, $(g) * $(r_pre));
		$(vbEff) = $(vb_post) * $(gb_post)/($(glk_post)+$(gb_post)+$(ga_post));
		$(dg) += DT * ($(muPP_basal) * $(r_pre)
		* ($(r_post) - {act_func('$(vbEff)')} ) - $(dg))/$(tau);
		$(g) += DT * $(dg); 
		//DT * $(muPP_basal) * $(r_pre)
		//* ($(r_post) - {act_func('$(vbEff)')} );
	""",
	"is_pre_spike_time_required": False,
	"is_post_spike_time_required": False
}

wu_model_pp_apical = {
	"class_name": "weight_update_model_pyr_to_pyr_back_def",
	"param_names": ["muPP_apical"],
	"var_name_types": [("g", "scalar")],
	"synapse_dynamics_code": f"""
	    $(addToInSyn, $(g) * $(r_pre));
	    
	    //$(g) += DT * $(muPP) * (
	    //	-$(va_exc_post)
	    //	+ $(g) * (1.0-$(va_exc_post)*$(va_exc_post))
	    //);
	    
	    //$(g) += DT * $(muPP_apical) * $(r_pre)
	    //  * ($(r_post) - $(va_exc_post));
	""",
	"is_pre_spike_time_required": False,
	"is_post_spike_time_required": False
}

wu_model_pinp = {
	"class_name": "weight_update_model_input_to_pyr",
	"param_names": ["muPINP","tau"],
	"var_name_types": [("g", "scalar"),("dg","scalar"),("vbEff","scalar")],
	"synapse_dynamics_code": f"""
		$(addToInSyn, $(g) * $(r_pre));
		$(vbEff) = $(vb_post) * $(gb_post)/($(glk_post)+$(gb_post)+$(ga_post));
		$(dg) += DT * ($(muPINP) * $(r_pre)
		* ($(r_post) - {act_func('$(vbEff)')} ) - $(dg))/$(tau);
		$(g) += DT * $(dg);
		//$(muPINP) * $(r_pre)
		//* ($(r_post) - {act_func('$(vbEff)')} );
	""",
	"is_pre_spike_time_required": False,
	"is_post_spike_time_required": False
}

wu_model_ip = {
    "class_name": "weight_update_model_pyr_to_int_def",
    "param_names": ["muIP","tau"],
    "var_name_types": [("g", "scalar"),("dg","scalar"),("vEff","scalar")],
    "synapse_dynamics_code": f"""
        $(addToInSyn, $(g) * $(r_pre));
        $(vEff) = $(v_post) * $(gd_post)/($(glk_post)+$(gd_post));
        $(dg) += DT * ($(muIP) * $(r_pre)
            * ($(r_post) - {act_func('$(vEff)')}) - $(dg))/$(tau);

        $(g) += DT * $(dg);
        //$(muIP) * $(r_pre)
        //* ($(r_post) - {act_func('$(vEff)')});
    """,
    "is_pre_spike_time_required": False,
    "is_post_spike_time_required": False
}

wu_model_ip_back = {
	"class_name": "weight_update_model_pyr_to_int_back_def",
	"param_names": [],
	"var_name_types": [("g","scalar")],
	"synapse_dynamics_code": f"""
		$(addToInSyn, $(g) * $(r_pre));
	""",
	"is_pre_spike_time_required": False,
	"is_post_spike_time_required": False
}

wu_model_pi = {
    "class_name": "weight_update_model_int_to_pyr_def",
    "param_names": ["muPI","tau"],
    "var_name_types": [("g", "scalar"),("dg","scalar")],
    "synapse_dynamics_code": """
        $(addToInSyn, $(g) * $(r_pre));
        $(dg) += DT * (-$(muPI) * $(r_pre)
            * $(va_post) - $(dg));
        $(g) += DT * $(dg);
        //$(muPI) * $(r_pre)
        //* $(va_post);
    """,
    "is_pre_spike_time_required": False,
    "is_post_spike_time_required": False
}