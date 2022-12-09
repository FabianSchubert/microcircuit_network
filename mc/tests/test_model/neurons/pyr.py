from ..utils import act_func, TH_COND_CODE, RESET_CODE

model_def = {
    "class_name": "pyr",
    "param_names": ["glk", "gb", "ga", "sigm_noise", "change_th"],
    "var_name_types": [("u", "scalar"), ("r", "scalar"),
                       ("r_last", "scalar"),
                       ("r_prev_last", "scalar"),
                       ("va_int", "scalar"), ("va_exc", "scalar"),
                       ("va", "scalar"),
                       ("vb", "scalar"),
                       ("t_last_spike", "scalar")],
    "additional_input_vars": [("Isyn_va_int", "scalar", 0.0),
                              ("Isyn_va_exc", "scalar", 0.0),
                              ("Isyn_vb", "scalar", 0.0)],
    "sim_code": f"""
                // Save the r_last of the previous time step
                $(r_prev_last) = $(r_last);
                
                $(vb) = $(Isyn_vb);
                $(va_int) = $(Isyn_va_int);
                $(va_exc) = $(Isyn_va_exc);
                $(va) = $(va_int) + $(va_exc);
                //relaxation
                $(u) += DT * ( -($(glk)+$(gb)+$(ga))*$(u)
                + $(gb)*$(vb)
                + $(ga)*$(va));
                //direct input
                //$(u) = ($(gb)*$(vb)+$(ga)*$(va))/($(glk)+$(gb)+$(ga));
                $(r) = {act_func('$(u)')};
                """,
    "threshold_condition_code": TH_COND_CODE,
    "reset_code": RESET_CODE,
    "is_auto_refractory_required": False

}

param_space = {
    "glk": 0.1,
    "ga": 0.8,
    "gb": 1.0,
    "sigm_noise": 0.0,
    "change_th": 0.0001
}

var_space = {
    "u": 0.0,
    "va": 0.0,
    "va_int": 0.0,
    "va_exc": 0.0,
    "vb": 0.0,
    "r": 0.0,
    "r_last": 1.0,
    "r_prev_last": 1.0,
    "t_last_spike": 0.0
}

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}