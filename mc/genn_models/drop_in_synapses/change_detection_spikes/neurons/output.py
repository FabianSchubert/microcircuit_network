from .utils import act_func, d_act_func

from genn_models.event_utils import convert_event_neur_dict, convert_event_neur_var_space_dict, convert_event_neur_param_space_dict

from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY

post_plast_vars = ["d_ra"]

model_def = {
    "class_name": "output",
    "param_names": ["tau_d_ra", "tau_prosp"],
    "var_name_types": [("r", "scalar"),
                       ("d_ra", "scalar"),
                       ("u", "scalar"),
                       ("u_prev", "scalar"),
                       ("u_prosp", "scalar"),
                       ("va", "scalar"),
                       ("vb", "scalar"),
                       ("ga", "scalar"),
                       ("b", "scalar", VarAccess_READ_ONLY), ("db", "scalar")],
    "additional_input_vars": [("Isyn_vb", "scalar", 0.0)],
    "sim_code": f"""
        $(u_prosp) = $(u) + ($(u) - $(u_prev)) * $(tau_prosp) / DT;

        $(r) = {act_func('$(u_prosp)')};

        $(va) = $(Isyn) - $(r);
        $(vb) = $(Isyn_vb) + $(b);

        $(u_prev) = $(u);
        $(u) += DT*($(ga) * $(va) + $(vb) - $(u));

        $(d_ra) += DT * ($(va) * {d_act_func('$(vb)')} - $(d_ra)) / $(tau_d_ra);
        //$(d_ra) = $(va) * {d_act_func('$(vb)')};

        $(db) += DT * $(d_ra);
    """
}

model_def = convert_event_neur_dict(model_def, post_plast_vars)

param_space = {
    "tau_d_ra": 10.,
    "tau_prosp": 1.
}

param_space = convert_event_neur_param_space_dict(param_space)

var_space = {
    "r": 0.0,
    "d_ra": 0.0,
    "u": 0.0,
    "u_prev": 0.0,
    "u_prosp": 0.0,
    "va": 0.0,
    "vb": 0.0,
    "ga": 0.1,
    "b": 0.0,
    "db": 0.0
}

var_space = convert_event_neur_var_space_dict(var_space, post_plast_vars)

mod_dat = {
    "model_def": model_def,
    "param_space": param_space,
    "var_space": var_space
}
