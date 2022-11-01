#! /usr/bin/env python3

from ..utils import act_func
from pygenn.genn_model import create_dpf_class

pyr_model = {
    "class_name": "pyr",
    "param_names": ["glk", "gb", "ga", "sigm_noise", "spike_interv"],
    "var_name_types": [("u", "scalar"), ("r", "scalar"),
                       ("r_last", "scalar"),
                       ("r_sec_last", "scalar"),
                       ("va_int", "scalar"), ("va_exc", "scalar"),
                       ("va", "scalar"),
                       ("vb", "scalar"),
                       ("t_last_spike", "scalar")],
    "additional_input_vars": [("Isyn_va_int", "scalar", 0.0),
                              ("Isyn_va_exc", "scalar", 0.0),
                              ("Isyn_vb", "scalar", 0.0)],
    "derived_params": [("DTSQRT",
                        create_dpf_class(lambda pars, dt: dt**.5)())],
    "sim_code": f"""
                $(vb) = $(Isyn_vb);
                $(va_int) = $(Isyn_va_int);
                $(va_exc) = $(Isyn_va_exc);
                $(va) = $(va_int) + $(va_exc);
                $(u) += DT * ( -($(glk)+$(gb)+$(ga))*$(u)
                + $(gb)*$(vb)
                + $(ga)*$(va));
                $(r) = {act_func('$(u)')};
                """,
    "threshold_condition_code": "$(t) - $(t_last_spike) > $(spike_interv)",
    "reset_code": """
                  $(r_sec_last) = $(r_last);
                  $(r_last) = $(r);
                  $(t_last_spike) = $(t);
                  """,
    "is_auto_refractory_required": False
}

output_model = {
    "class_name": "output",
    "param_names": ["glk", "gb", "ga", "sigm_noise", "pop_size",
                    "spike_interv"],
    "var_name_types": [("u", "scalar"), ("r", "scalar"),
                       ("vb", "scalar"), ("gnudge", "scalar"),
                       ("vnudge", "scalar"), ("idx_dat", "int"),
                       ("r_last", "scalar"), ("r_sec_last", "scalar"),
                       ("t_last_spike", "scalar")],
    "additional_input_vars": [("Isyn_vb", "scalar", 0.0)],
    "derived_params": [("DTSQRT",
                        create_dpf_class(lambda pars, dt: dt**.5)())],
    "sim_code": f"""
                $(vb) = $(Isyn_vb);
                if($(idx_dat) < $(size_t_sign)){{
                    if($(t)>=$(t_sign)[$(idx_dat)]*DT){{
                        $(vnudge) = $(u_trg)[$(id)+$(idx_dat)*$(size_u_trg)];
                        $(idx_dat)++;
                    }}
                }}
                $(u) += DT * (-($(glk)+$(gb)+$(gnudge))*$(u)
                + $(gb)*$(vb)
                + $(gnudge)*$(vnudge));
                $(r) = {act_func('$(u)')};
                """,
    "threshold_condition_code": "$(t) - $(t_last_spike) > $(spike_interv)",
    "reset_code": """
                  $(r_sec_last) = $(r_last);
                  $(r_last) = $(r);
                  $(t_last_spike) = $(t);
                  """,
    "extra_global_params": [("u_trg", "scalar*"),
                            ("size_u_trg", "int"), 
                            ("t_sign", "int*"),
                            ("size_t_sign", "int")],
    "is_auto_refractory_required": False
}

int_model = {
    "class_name": "int",
    "param_names": ["glk", "gd", "gsom", "spike_interv"],
    "var_name_types": [("u", "scalar"), ("v", "scalar"), ("r", "scalar"),
                       ("r_last", "scalar"), ("r_sec_last", "scalar"),
                       ("t_last_spike", "scalar")],
    "additional_input_vars": [("u_td", "scalar", 0.0)],
    "sim_code": f"""
                $(v) = $(Isyn);
                $(u) += DT * ( -$(glk)*$(u)
                + $(gd)*( $(v)-$(u) )
                + $(gsom)*( $(u_td) - $(u) ));
                $(r) = {act_func('$(u)')};
                """,
    "threshold_condition_code": "$(t) - $(t_last_spike) > $(spike_interv)",
    "reset_code": """
                  $(r_sec_last) = $(r_last);
                  $(r_last) = $(r);
                  $(t_last_spike) = $(t);
                  """,
    "is_auto_refractory_required": False
}

input_model = {
    "class_name": "input",
    "param_names": ["pop_size", "spike_interv"],
    "var_name_types": [("r", "scalar"), ("idx_dat", "int"),
                       ("r_last", "scalar"), ("r_sec_last", "scalar"),
                       ("t_last_spike", "scalar")],
    "sim_code": """
    if($(idx_dat) < $(size_t_sign)){
        if($(t)>=$(t_sign)[$(idx_dat)]*DT){
            $(r) = $(u)[$(id)+$(idx_dat)*$(size_u)];
            $(idx_dat)++;
        }
    }""",
    "threshold_condition_code": "$(t) - $(t_last_spike) > $(spike_interv)",
    "reset_code": """
                  $(r_sec_last) = $(r_last);
                  $(r_last) = $(r);
                  $(t_last_spike) = $(t);
                  """,
    "extra_global_params": [("u", "scalar*"),
                            ("size_u", "int"),
                            ("t_sign", "int*"),
                            ("size_t_sign", "int")],
    "is_auto_refractory_required": False
}
