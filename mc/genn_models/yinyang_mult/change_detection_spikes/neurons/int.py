from ..utils import act_func, d_act_func, TH_COND_CODE, RESET_CODE

from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY

model_def = {
    "class_name": "int",
    "param_names": ["th", "ga", "tau_va", "tau_d_ra"],
    "var_name_types": [("r", "scalar"), ("r_prev", "scalar"),
                       ("r_prev_prev", "scalar"),
                       ("d_ra", "scalar"), ("d_ra_prev", "scalar"),
                       ("d_ra_prev_prev", "scalar"),
                       ("r_eff", "scalar"), ("r_eff_prev", "scalar"),
                       ("r_eff_prev_prev", "scalar"),
                       ("u", "scalar"),
                       ("va", "scalar"),
                       ("vb", "scalar"),
                       ("b", "scalar", VarAccess_READ_ONLY), ("db", "scalar")],
    "additional_input_vars": [("u_td", "scalar", 0.0)],
    "sim_code": f"""

        $(d_ra_prev) = $(d_ra);

        $(vb) = $(Isyn) + $(b);

        $(r_eff) = {act_func('$(vb)')};

        $(va) = $(u_td) - $(r_eff);

        $(u) += DT*($(ga) * $(va) + $(vb) - $(u));
        //$(u) = $(ga) * $(va) + $(vb);

        $(r) = {act_func('$(u)')};
        
        $(d_ra) += DT * ($(va) * {d_act_func('$(vb)')} - $(d_ra)) / $(tau_d_ra);

        $(db) += DT * $(d_ra);
    """,
    "threshold_condition_code": TH_COND_CODE,
    "reset_code": RESET_CODE,
    "is_auto_refractory_required": False
}

param_space = {
    "th": 1e-3,
    "ga": 0.0,
    "tau_va": 1.,
    "tau_d_ra": 15.
}

var_space = {
    "r": 0.0,
    "r_prev": 0.0,
    "r_prev_prev": 0.0,
    "r_eff": 0.0,
    "r_eff_prev": 0.0,
    "r_eff_prev_prev": 0.0,
    "d_ra": 0.0,
    "d_ra_prev": 0.0,
    "d_ra_prev_prev": 0.0,
    "u": 0.0,
    "va": 0.0,
    "vb": 0.0,
    "b": 0.0,
    "db": 0.0
}

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}