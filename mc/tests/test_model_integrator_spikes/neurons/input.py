from ..utils import act_func_input

#TH_COND_CODE_INPUT = "$(t)-$(t_last_spike) >= 1./$(r)"
TH_COND_CODE_INPUT = "$(gennrand_uniform) <= $(r)*DT"
#TH_COND_CODE_INPUT = "$(u) >= $(u_th)"

#RESET_CODE_INPUT = "$(t_last_spike) = $(t);"
RESET_CODE_INPUT = ""
#RESET_CODE_INPUT = "$(u) = $(u_r);"

model_def = {
    "class_name": "input",
    "param_names": ["pop_size", "u_th", "u_r"],
    "var_name_types": [("r", "scalar"), ("idx_dat", "int"),
                       ("t_last_spike", "scalar")],
    "sim_code": f"""
    if($(idx_dat) < $(size_t_sign)){{
        if($(t)>=$(t_sign)[$(idx_dat)]*DT){{
            $(r) = {act_func_input('$(u)[$(id)+$(batch)*$(size_u)+$(idx_dat)*$(size_u)*$(batch_size)]')};
            $(idx_dat)++;
        }}
    }}""",
    "threshold_condition_code": TH_COND_CODE_INPUT,
    "reset_code": RESET_CODE_INPUT,
    "extra_global_params": [("u", "scalar*"),
                            ("size_u", "int"),
                            ("batch_size", "int"),
                            ("t_sign", "int*"),
                            ("size_t_sign", "int")],
    "is_auto_refractory_required": False
}

param_space = {
    "pop_size": None,
    "u_th": 1.0,
    "u_r": 0.0
}

var_space = {
    "r": 0.0,
    "idx_dat": 0,
    "t_last_spike": 0.0
}

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}

