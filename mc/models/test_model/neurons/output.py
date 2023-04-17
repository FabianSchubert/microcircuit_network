from ..utils import act_func, TH_COND_CODE, RESET_CODE

model_def = {
    "class_name": "output",
    "param_names": [],
    "var_name_types": [("r", "scalar"), ("u", "scalar"), ("vb", "scalar")],
    "additional_input_vars": [("Isyn_vb", "scalar", 0.0)],
    "sim_code": f"""
                $(vb) = $(Isyn_vb);

                $(u) += DT * (
                $(vb)
                + $(Isyn)
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
    "r": 0.0,
    "vb": 0.0
}

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}