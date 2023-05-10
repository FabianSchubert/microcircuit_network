from ..utils import act_func, TH_COND_CODE, RESET_CODE

from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY

model_def = {
    "class_name": "input",
    "param_names": [],
    "var_name_types": [("r", "scalar")
                       ("r_eff", "scalar"),
                       ("d_ra", "scalar"),
                       ("u", "scalar"),
                       ("b", "scalar", VarAccess_READ_ONLY), ("db", "scalar")],
    "sim_code": f"""
        $(u) = $(Isyn);
        $(r) = $(u);
    """,
    "threshold_condition_code": TH_COND_CODE,
    "reset_code": RESET_CODE,
    "is_auto_refractory_required": False
}

param_space = {}

var_space = {
    "r": 0.0,
    "r_eff": 0.0,
    "d_ra": 0.0,
    "u": 0.0,
    "b": 0.0,
    "db": 0.0
}

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}
