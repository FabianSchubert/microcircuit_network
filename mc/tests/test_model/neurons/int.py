from ..utils import act_func, TH_COND_CODE, RESET_CODE

model_def = {
    "class_name": "int",
    "param_names": ["glk", "gd", "gsom", "change_th"],
    "var_name_types": [("u", "scalar"), ("v", "scalar"), ("r", "scalar"),
                       ("r_last", "scalar"), ("r_prev_last", "scalar"),
                       ("t_last_spike", "scalar")],
    "additional_input_vars": [("u_td", "scalar", 0.0)],
    "sim_code": f"""
                $(r_prev_last) = $(r_last);
                
                $(v) = $(Isyn);
                // relaxation
                $(u) += DT * ( -$(glk)*$(u)
                + $(gd)*( $(v)-$(u) )
                + $(gsom)*( $(u_td) - $(u) ));
                // direct input
                //$(u) = ($(gd)*$(v)+$(gsom)*$(u_td))/($(glk)+$(gd)+$(gsom));
                $(r) = {act_func('$(u)')};
                """,
    "threshold_condition_code": TH_COND_CODE,
    "reset_code": RESET_CODE,
    "is_auto_refractory_required": False

}

param_space = {
    "glk": 0.1,
    "gd": 1.0,
    "gsom": 0.8,
    "change_th": 0.0001
}

var_space = {
    "u": 0.0,
    "v": 0.0,
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