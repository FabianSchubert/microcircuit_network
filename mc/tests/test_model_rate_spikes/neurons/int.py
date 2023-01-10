from ..utils import act_func, TH_COND_CODE, RESET_CODE

model_def = {
    "class_name": "int",
    "param_names": ["glk", "gd", "gsom", "u_th", "u_r"],
    "var_name_types": [("u", "scalar"), ("v", "scalar"),
                       ("vEff", "scalar"),
                       ("r", "scalar"),
                       ("t_last_spike", "scalar")],
    "additional_input_vars": [("u_td", "scalar", 0.0)],
    "sim_code": f"""                
                $(v) = $(Isyn);
                $(vEff) = $(v) * $(gd)/($(glk)+$(gd));
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
    "u_th": 1.0,
    "u_r": 0.0
}

var_space = {
    "u": 0.0,
    "v": 0.0,
    "vEff": 0.0,
    "r": 0.0,
    "t_last_spike": 0.0
}

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}