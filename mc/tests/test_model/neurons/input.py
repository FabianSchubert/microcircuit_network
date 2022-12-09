from ..utils import TH_COND_CODE, RESET_CODE

model_def = {
    "class_name": "input",
    "param_names": ["pop_size", "change_th"],
    "var_name_types": [("r", "scalar"), ("idx_dat", "int"),
                       ("r_last", "scalar"), ("r_prev_last", "scalar"),
                       ("t_last_spike", "scalar")],
    "sim_code": """
    $(r_prev_last) = $(r_last);
           
    if($(idx_dat) < $(size_t_sign)){
        if($(t)>=$(t_sign)[$(idx_dat)]*DT){
            $(r) = $(u)[$(id)+$(idx_dat)*$(size_u)];
            $(idx_dat)++;
        }
    }""",
    "threshold_condition_code": TH_COND_CODE,
    "reset_code": RESET_CODE,
    "extra_global_params": [("u", "scalar*"),
                            ("size_u", "int"),
                            ("t_sign", "int*"),
                            ("size_t_sign", "int")],
    "is_auto_refractory_required": False

}

param_space = {
    "pop_size": None,
    "change_th": 0.0001
}

var_space = {
    "r": 0.0,
    "idx_dat": 0,
    "r_last": 1.0,
    "r_prev_last": 1.0,
    "t_last_spike": 0.0
}

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}

