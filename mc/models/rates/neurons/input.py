from ..utils import TH_COND_CODE, RESET_CODE

model_def = {
    "class_name": "input",
    "param_names": ["pop_size"],
    "var_name_types": [("r", "scalar"), ("idx_dat", "int")],
    "sim_code": """           
    if($(idx_dat) < $(size_t_sign)){
        if($(t)>=$(t_sign)[$(idx_dat)]*DT){
            $(r) = $(u)[$(id)+$(batch)*$(size_u)+$(idx_dat)*$(size_u)*$(batch_size)];
            $(idx_dat)++;
        }
    }""",
    "threshold_condition_code": TH_COND_CODE,
    "reset_code": RESET_CODE,
    "extra_global_params": [("u", "scalar*"),
                            ("size_u", "int"),
                            ("batch_size", "int"),
                            ("t_sign", "int*"),
                            ("size_t_sign", "int")],
    "is_auto_refractory_required": False
}

param_space = {
    "pop_size": None
}

var_space = {
    "r": 0.0,
    "idx_dat": 0
}

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}

