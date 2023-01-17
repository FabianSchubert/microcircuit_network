from ..utils import act_func, TH_COND_CODE, RESET_CODE


model_def = {
    "class_name": "output",
    "param_names": ["glk", "gb", "ga", "sigm_noise", "pop_size"],
    "var_name_types": [("u", "scalar"), ("r", "scalar"),
                       ("vb", "scalar"), ("vbEff", "scalar"),
                       ("gnudge", "scalar"), ("vnudge", "scalar"), ("idx_dat", "int")],
    "additional_input_vars": [("Isyn_vb", "scalar", 0.0)],
    "sim_code": f"""
                $(vb) = $(Isyn_vb);
                $(vbEff) = $(vb) * $(gb)/($(glk)+$(gb)+$(ga));
                
                if($(idx_dat) < $(size_t_sign)){{
                    if($(t)>=$(t_sign)[$(idx_dat)]*DT){{
                        $(vnudge) = $(u_trg)[$(id)+$(batch)*$(size_u_trg)+$(idx_dat)*$(size_u_trg)*$(batch_size)];
                        $(idx_dat)++;
                    }}
                }}
                
                //relaxation
                $(u) += DT * (-($(glk)+$(gb)+$(gnudge))*$(u)
                + $(gb)*$(vb)
                + $(gnudge)*$(vnudge));
                
                $(r) = {act_func('$(u)')};
                """,
    "threshold_condition_code": TH_COND_CODE,
    "reset_code": RESET_CODE,
    "extra_global_params": [("u_trg", "scalar*"),
                            ("size_u_trg", "int"),
                            ("batch_size", "int"),
                            ("t_sign", "int*"),
                            ("size_t_sign", "int")],
    "is_auto_refractory_required": False
}
param_space = {
    "glk": 0.1,
    "gb": 1.0,
    "ga": 0.0,
    "sigm_noise": 0.0,
    "pop_size": None
}

var_space = {
    "u": 0.0,
    "r": 0.0,
    "vb": 0.0,
    "vbEff": 0.0,
    "gnudge": 0.8,
    "vnudge": 0.0,
    "idx_dat": 0
}

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}