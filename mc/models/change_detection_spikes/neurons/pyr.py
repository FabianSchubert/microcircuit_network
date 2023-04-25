from ..utils import act_func, TH_COND_CODE, RESET_CODE

model_def = {
   "class_name": "pyr",
    "param_names": ["th", "ga"],
    "var_name_types": [("r", "scalar"), ("r_prev", "scalar"),
                       ("r_prev_prev", "scalar"),
                       ("r_eff", "scalar"), ("r_eff_prev", "scalar"),
                       ("r_eff_prev_prev", "scalar"),
                       ("u", "scalar"),
                       ("va", "scalar"),
                       ("va_exc", "scalar"), ("va_int", "scalar"),
                       ("vb", "scalar")],
    "additional_input_vars": [("Isyn_va_int", "scalar", 0.0),
                              ("Isyn_va_exc", "scalar", 0.0),
                              ("Isyn_vb", "scalar", 0.0)],
    "sim_code": f"""
        $(va_exc) = $(Isyn_va_exc);
        $(va_int) = $(Isyn_va_int);        
        $(va) = $(va_exc) + $(va_int);
        
        $(vb) = $(Isyn_vb);
 
        //const scalar u_prev = $(u);
                
        //$(u) += DT*($(ga) * $(va) + $(vb) - $(u));
        $(u) = $(ga) * $(va) + $(vb);
        
        $(r) = {act_func('$(u)', 0.025)};
        $(r_eff) = {act_func('$(vb)', 0.025)};
    """,
    "threshold_condition_code": TH_COND_CODE,
    "reset_code": RESET_CODE,
    "is_auto_refractory_required": False
}

param_space = {
    "th": 0.005,
    "ga": 0.0
}

var_space = {
    "r": 0.0,
    "r_prev": 0.0,
    "r_prev_prev": 0.0,
    "r_eff": 0.0,
    "r_eff_prev": 0.0,
    "r_eff_prev_prev": 0.0,
    "u": 0.0,
    "va_exc": 0.0,
    "va_int": 0.0,
    "va": 0.0,
    "vb": 0.0
}

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}