from ..utils import act_func, TH_COND_CODE, RESET_CODE

RESET_CODE_INT = f"""
{RESET_CODE}
$(ca) += 1.0 / $(tau_ca);
$(t_last_spike) = $(t);
"""


model_def = {
    "class_name": "int",
    "param_names": ["glk", "gd", "gsom", "u_th", "u_r", "tau_ca", "tau_v", "tau_bias", "gain_rEff",
                    "r_targ"],
    "var_name_types": [("u", "scalar"), ("v", "scalar"),
                       ("va", "scalar"),
                       ("vEff", "scalar"),
                       ("rEff", "scalar"),
                       #("r", "scalar"),
                       ("t_last_spike", "scalar"),
                       ("ca", "scalar"),
                       ("I_bias", "scalar")],
    "additional_input_vars": [("u_td", "scalar", 0.0)],
    "sim_code": f"""                
                $(v) += DT * ($(Isyn) - $(v)) / $(tau_v);
                $(vEff) = $(v) * $(gd);// /($(glk)+$(gd));
                
                $(rEff) = $(gain_rEff) * max(0.0, $(vEff) - $(I_bias)) / ($(u_th) - $(u_r));
                
                $(va) += DT * ($(u_td) - $(va)) / $(tau_v);
                
                //$(I_bias) += DT * ($(vEff) - $(I_bias)) / $(tau_bias);
                
                // integration
                $(u) += DT * (
                + $(vEff)
                + $(gsom)*$(va)
                - $(I_bias)
                );
                
                $(u) = max($(u), $(u_r));
                
                $(ca) += -DT * $(ca) / $(tau_ca);
                """,
    "threshold_condition_code": TH_COND_CODE,
    "reset_code": RESET_CODE_INT,
    "is_auto_refractory_required": False

}

param_space = {
    "glk": 0.1,
    "gd": 1.0,
    "gsom": 0.1,
    "u_th": 1.0,
    "u_r": 0.0,
    "tau_ca": 50.0,
    "tau_v": 50.0,
    "tau_bias": 5000.0,
    "gain_rEff": 2.0,
    "r_targ": 0.1
}

var_space = {
    "u": 0.0,
    "v": 0.0,
    "va": 0.0,
    "vEff": 0.0,
    "rEff": 0.0,
    #"r": 0.0,
    "t_last_spike": 0.0,
    "ca": 0.0,
    "I_bias": 0.0
}

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}