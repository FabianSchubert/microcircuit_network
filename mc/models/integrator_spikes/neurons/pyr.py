from ..utils import act_func, TH_COND_CODE, RESET_CODE

RESET_CODE_PYR = f"""
{RESET_CODE}
$(ca) += 1.0 / $(tau_ca);
$(t_last_spike) = $(t);
"""

model_def = {
    "class_name": "pyr",
    "param_names": ["glk", "gb", "ga", "u_th", "u_r", "tau_ca", "r_targ", "tau_v", "tau_bias"],
    "var_name_types": [("u", "scalar"),# ("r", "scalar"),
                       ("va_int", "scalar"), ("va_exc", "scalar"),
                       ("va", "scalar"),
                       ("vb", "scalar"),
                       ("vbEff", "scalar"),
                       ("rEff", "scalar"),
                       ("t_last_spike", "scalar"),
                       ("I_bias", "scalar"),
                       ("ca", "scalar")],
    "additional_input_vars": [("Isyn_va_int", "scalar", 0.0),
                              ("Isyn_va_exc", "scalar", 0.0),
                              ("Isyn_vb", "scalar", 0.0)],
    "sim_code": f"""
                $(vb) += DT * ($(Isyn_vb) - $(vb)) / $(tau_v);
                $(vbEff) = $(vb) * $(gb);// /($(glk)+$(gb)+$(ga));
                
                $(rEff) = max($(vbEff) - $(I_bias), 0.0) / ($(u_th) - $(u_r));
                
                $(va_int) = $(Isyn_va_int);
                $(va_exc) = $(Isyn_va_exc);
                $(va) += DT * ($(va_int) + $(va_exc) - $(va)) / $(tau_v);
                
                //$(I_bias) += DT * ($(vbEff) - $(I_bias)) / $(tau_bias);
                
                $(u) += DT * (
                + $(vbEff)
                + $(ga)*$(va)
                - $(I_bias));
                
                $(u) = max($(u), $(u_r));
                
                $(ca) += -DT * $(ca) / $(tau_ca);
                """,
    "threshold_condition_code": TH_COND_CODE,
    "reset_code": RESET_CODE_PYR,
    "is_auto_refractory_required": False

}

param_space = {
    "glk": 0.1,
    "ga": 0.2,
    "gb": 1.0,
    "u_th": 1.0,
    "u_r": 0.0,
    "tau_ca": 50.0,
    "tau_bias": 5000.0,
    "tau_v": 50.0,
    "r_targ": 0.1
}

var_space = {
    "u": 0.0,
    "va": 0.0,
    "va_int": 0.0,
    "va_exc": 0.0,
    "vb": 0.0,
    "vbEff": 0.0,
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