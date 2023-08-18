from ..utils import act_func, TH_COND_CODE, RESET_CODE

model_def = {
    "class_name": "int",
    "param_names": [],
    "var_name_types": [("u", "scalar"), ("v", "scalar"), ("r", "scalar")],
    "additional_input_vars": [("u_td", "scalar", 0.0)],
    "sim_code": f"""
                $(v) = $(Isyn);
                
                $(u) += DT * (
                + $(v) - $(u)
                + $(u_td) - $(u)
                );
                
                $(r) = {act_func('$(u)')};
                """,
    "threshold_condition_code": TH_COND_CODE,
    "reset_code": RESET_CODE,
    "is_auto_refractory_required": False

}

param_space = {
}

var_space = {
    "u": 0.0,
    "v": 0.0,
    "r": 0.0
}

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}