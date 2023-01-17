from ..utils import act_func, TH_COND_CODE, RESET_CODE

model_def = {
    "class_name": "pyr",
    "param_names": ["glk", "gb", "ga", "u_th", "u_r"],
    "var_name_types": [("u", "scalar"), ("r", "scalar"),
                       ("va_int", "scalar"), ("va_exc", "scalar"),
                       ("va", "scalar"),
                       ("vb", "scalar"),
                       ("vbEff", "scalar"),
                       ("t_last_spike", "scalar")],
    "additional_input_vars": [("Isyn_va_int", "scalar", 0.0),
                              ("Isyn_va_exc", "scalar", 0.0),
                              ("Isyn_vb", "scalar", 0.0)],
    "sim_code": f"""
                $(vb) = $(Isyn_vb);
                $(vbEff) = $(vb) * $(gb)/($(glk)+$(gb)+$(ga));
                $(va_int) = $(Isyn_va_int);
                $(va_exc) = $(Isyn_va_exc);
                $(va) = $(va_int) + $(va_exc);
                
                $(u) += DT * ( -($(glk)+$(gb)+$(ga))*$(u)
                + $(gb)*$(vb)
                + $(ga)*$(va));
                $(r) = {act_func('$(u)')};
                """,
    "threshold_condition_code": TH_COND_CODE,
    "reset_code": RESET_CODE,
    "is_auto_refractory_required": False

}

param_space = {
    "glk": 0.1,
    "ga": 0.1,
    "gb": 1.0,
    "u_th": 0.1,
    "u_r": 0.0
}

var_space = {
    "u": 0.0,
    "va": 0.0,
    "va_int": 0.0,
    "va_exc": 0.0,
    "vb": 0.0,
    "vbEff": 0.0,
    "r": 0.0,
    "t_last_spike": 0.0
}

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}