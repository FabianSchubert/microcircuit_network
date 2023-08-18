from ..utils import act_func, TH_COND_CODE, RESET_CODE

model_def = {
    "class_name": "pyr",
    "param_names": [],
    "var_name_types": [("u", "scalar"), ("r", "scalar"),
                       ("vb", "scalar"), ("va", "scalar"),
                       ("va_int", "scalar"), ("va_exc", "scalar")],
    "additional_input_vars": [("Isyn_va_int", "scalar", 0.0),
                              ("Isyn_va_exc", "scalar", 0.0),
                              ("Isyn_vb", "scalar", 0.0)],
    "sim_code": f"""
                $(vb) = $(Isyn_vb);
                $(va_int) = $(Isyn_va_int);
                $(va_exc) = $(Isyn_va_exc);
                $(va) = $(va_int) + $(va_exc);

                $(u) += DT * (
                $(vb)
                + $(va)
                 -$(u));
     
                $(r) = {act_func('$(u)')};
                """,
    "threshold_condition_code": TH_COND_CODE,
    "reset_code": RESET_CODE
}

param_space = {
}

var_space = {
    "u": 0.0,
    "va": 0.0,
    "va_int": 0.0,
    "va_exc": 0.0,
    "vb": 0.0,
    "r": 0.0
}

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}