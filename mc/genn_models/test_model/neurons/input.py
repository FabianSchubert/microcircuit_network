from ..utils import TH_COND_CODE, RESET_CODE, act_func

model_def = {
    "class_name": "input",
    "param_names": [],
    "var_name_types": [("r", "scalar"), ("u", "scalar")],
    "sim_code": """
    $(u) += DT * ($(Isyn) - $(u));
    $(r) = $(u);
    """,
    "threshold_condition_code": TH_COND_CODE,
    "reset_code": RESET_CODE
}

param_space = {
}

var_space = {
    "r": 0.0,
    "u": 0.0
}

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}

