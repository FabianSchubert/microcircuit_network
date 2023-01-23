from ..utils import act_func, TH_COND_CODE, RESET_CODE

RESET_CODE_OUT = f"""
{RESET_CODE}
$(ca) += 1.0 / $(tau_ca);
"""

model_def = {
    "class_name": "output",
    "param_names": ["glk", "gb", "ga", "pop_size", "u_th", "u_r", "tau_ca"],
    "var_name_types": [("u", "scalar"),# ("r", "scalar"),
                       ("vb", "scalar"),
                       ("vbEff", "scalar"),
                       ("rEff", "scalar"),
                       ("gnudge", "scalar"),
                       ("vnudge", "scalar"), ("idx_dat", "int"),
                       ("t_last_spike", "scalar"),
                       ("ca", "scalar")],
    "additional_input_vars": [("Isyn_vb", "scalar", 0.0)],
    "sim_code": f"""                
                $(vb) = $(Isyn_vb);
                
                $(vbEff) = $(vb) * $(gb);// /($(glk)+$(gb)+$(ga));
                
                if($(idx_dat) < $(size_t_sign)){{
                    if($(t)>=$(t_sign)[$(idx_dat)]*DT){{
                        $(vnudge) = $(u_trg)[$(id)+$(batch)*$(size_u_trg)+$(idx_dat)*$(size_u_trg)*$(batch_size)];
                        $(idx_dat)++;
                    }}
                }}
                
                $(rEff) = max($(vb),0.0) * $(gb) / ($(u_th) - $(u_r));
                
                //integration
                $(u) += DT * (
                + $(vbEff)
                + $(gnudge)*($(vnudge) - $(ca))
                );
                
                $(u) = max($(u), $(u_r));
                                
                $(ca) += -DT * $(ca) / $(tau_ca);
                """,
    "threshold_condition_code": TH_COND_CODE,
    "reset_code": RESET_CODE_OUT,
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
    "pop_size": None,
    "u_th": 1.0,
    "u_r": 0.0,
    "tau_ca": 50.0
}

var_space = {
    "u": 0.0,
    #"r": 0.0,
    "vb": 0.0,
    "vbEff": 0.0,
    "rEff": 0.0,
    "gnudge": 0.8,
    "vnudge": 0.0,
    "idx_dat": 0,
    "t_last_spike": 0.0,
    "ca": 0.0
}

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}