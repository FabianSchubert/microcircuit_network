from ..utils import act_func, TH_COND_CODE, RESET_CODE

RESET_CODE_INT = f"""
{RESET_CODE}
$(ca) += 1.0 / $(tau_ca);
"""


model_def = {
    "class_name": "int",
    "param_names": ["glk", "gd", "gsom", "u_th", "u_r", "tau_ca"],
    "var_name_types": [("u", "scalar"), ("v", "scalar"),
                       ("va", "scalar"),
                       ("vEff", "scalar"),
                       ("rEff", "scalar"),
                       #("r", "scalar"),
                       ("t_last_spike", "scalar"),
                       ("ca", "scalar")],
    "additional_input_vars": [("u_td", "scalar", 0.0)],
    "sim_code": f"""                
                $(v) = $(Isyn);
                $(vEff) = $(v) * $(gd);// /($(glk)+$(gd));
                
                $(rEff) = max($(v),0.0) * $(gd) / ($(u_th) - $(u_r) - $(gsom));
                
                $(va) = $(u_td);
                
                // integration
                $(u) += DT * (
                + $(vEff)
                + $(gsom)*$(va)
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
    "gsom": 0.5,
    "u_th": 1.0,
    "u_r": 0.0,
    "tau_ca": 50.0
}

var_space = {
    "u": 0.0,
    "v": 0.0,
    "va": 0.0,
    "vEff": 0.0,
    "rEff": 0.0,
    #"r": 0.0,
    "t_last_spike": 0.0,
    "ca": 0.0
}

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}