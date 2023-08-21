from .utils import act_func, d_act_func

from genn_models.utils import convert_neuron_mod_data_cont_to_event

from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY

from ..settings import mod_type

post_plast_vars = ["d_ra", "va"]

model_def = {
    "class_name": "pyr",
    "param_names": ["ga", "tau_d_ra", "tau_prosp"],
    "var_name_types": [("r", "scalar"),
                       ("d_ra", "scalar"),
                       ("u", "scalar"),
                       ("u_prev", "scalar"),
                       ("u_prosp", "scalar"),
                       ("va", "scalar"),
                       ("va_exc", "scalar"), ("va_int", "scalar"),
                       ("vb", "scalar"),
                       ("b", "scalar", VarAccess_READ_ONLY), ("db", "scalar")],
    "additional_input_vars": [("Isyn_va_int", "scalar", 0.0),
                              ("Isyn_va_exc", "scalar", 0.0),
                              ("Isyn_vb", "scalar", 0.0)],
    "sim_code": f"""
        $(u_prosp) = $(u) + ($(u) - $(u_prev)) * $(tau_prosp) / DT;

        $(r) = {act_func('$(u_prosp)')};

        $(va_exc) = $(Isyn_va_exc);
        $(va_int) = $(Isyn_va_int);
        $(va) = $(va_exc) + $(va_int);

        $(vb) = $(Isyn_vb) + $(b);

        $(u_prev) = $(u);
        $(u) += DT*($(ga) * $(va) + $(vb) - $(u));

        $(d_ra) += DT * ($(va) * {d_act_func('$(vb)')} - $(d_ra)) / $(tau_d_ra);
        //$(d_ra) = $(va) * {d_act_func('$(vb)')};

        $(db) += DT * $(d_ra);
    """
}

param_space = {
    "ga": 0.1,
    "tau_d_ra": 10.,
    "tau_prosp": 1.
}

var_space = {
    "r": 0.0,
    "d_ra": 0.0,
    "u": 0.0,
    "u_prev": 0.0,
    "u_prosp": 0.0,
    "va_exc": 0.0,
    "va_int": 0.0,
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

if mod_type == "event":
    mod_dat = convert_neuron_mod_data_cont_to_event(mod_dat, post_plast_vars)
