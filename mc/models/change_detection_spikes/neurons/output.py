from ..utils import act_func, TH_COND_CODE, RESET_CODE

model_def = {
    "class_name": "output",
    "param_names": ["th"],
    "var_name_types": [("r", "scalar"), ("r_prev", "scalar"),
                       ("r_prev_prev", "scalar"),
                       ("r_eff", "scalar"), ("r_eff_prev", "scalar"),
                       ("r_eff_prev_prev", "scalar"),
                       ("r_target", "scalar"),
                       ("u", "scalar"), ("delta", "scalar"),
                       ("va", "scalar"),
                       ("vb", "scalar"),
                       ("ga", "scalar")],
    "additional_input_vars": [("Isyn_vb", "scalar", 0.0)],
    "sim_code": f"""
        $(r_target) = $(Isyn);
        $(va) += DT * ($(r_target) - $(r) - $(va));

        $(vb) += DT * ($(Isyn_vb) - $(vb));

        $(u) = $(ga) * $(va) + $(vb);

        $(r) = {act_func('$(u)')};
        $(r_eff) = {act_func('$(vb)')};
    """,
    "threshold_condition_code": TH_COND_CODE,
    "reset_code": RESET_CODE,
    "is_auto_refractory_required": False
}

param_space = {
    "th": 0.05
}

var_space = {
    "r": 0.0,
    "r_prev": 0.0,
    "r_prev_prev": 0.0,
    "r_eff": 0.0,
    "r_eff_prev": 0.0,
    "r_eff_prev_prev": 0.0,
    "r_target": 0.0,
    "u": 0.0,
    "delta": 0.0,
    "va": 0.0,
    "vb": 0.0,
    "ga": 0.1
}

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}