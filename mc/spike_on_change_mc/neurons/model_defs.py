"""
Neuron model definitions with keywords stored
in dicts.
"""

from pygenn.genn_model import create_dpf_class

from ..utils import act_func

TH_COND_CODE = "abs($(r)-$(r_last)) >= $(change_th)"
RESET_CODE = "$(r_last) = $(r);"

pyr_model = {
    "class_name": "pyr",
    "param_names": ["glk", "gb", "ga", "sigm_noise", "change_th"],
    "var_name_types": [("u", "scalar"), ("r", "scalar"),
                       ("r_last", "scalar"),
                       # ("r_sec_last", "scalar"),
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
                //relaxation
                $(u) += DT * ( -($(glk)+$(gb)+$(ga))*$(u)
                + $(gb)*$(vb)
                + $(ga)*$(va));
                //direct input
                //$(u) = ($(gb)*$(vb)+$(ga)*$(va))/($(glk)+$(gb)+$(ga));
                $(r) = {act_func('$(u)')};
                """,
    "threshold_condition_code": TH_COND_CODE,
    "reset_code": RESET_CODE,
    "is_auto_refractory_required": False
}

output_model = {
    "class_name": "output",
    "param_names": ["glk", "gb", "ga", "sigm_noise", "pop_size",
                    "change_th"],
    "var_name_types": [("u", "scalar"), ("r", "scalar"),
                       ("vb", "scalar"), ("gnudge", "scalar"),
                       ("vnudge", "scalar"), ("idx_dat", "int"),
                       ("r_last", "scalar"),# ("r_sec_last", "scalar"),
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
                //relaxation
                $(u) += DT * (-($(glk)+$(gb)+$(gnudge))*$(u)
                + $(gb)*$(vb)
                + $(gnudge)*$(vnudge));
                //direct input
                //$(u) = ($(gb)*$(vb)+$(gnudge)*$(vnudge))/($(glk)+$(gb)+$(gnudge));
                $(r) = {act_func('$(u)')};
                """,
    "threshold_condition_code": TH_COND_CODE,
    "reset_code": RESET_CODE,
    "extra_global_params": [("u_trg", "scalar*"),
                            ("size_u_trg", "int"),
                            ("t_sign", "int*"),
                            ("size_t_sign", "int")],
    "is_auto_refractory_required": False
}

int_model = {
    "class_name": "int",
    "param_names": ["glk", "gd", "gsom", "change_th"],
    "var_name_types": [("u", "scalar"), ("v", "scalar"), ("r", "scalar"),
                       ("r_last", "scalar"),# ("r_sec_last", "scalar"),
                       ("t_last_spike", "scalar")],
    "additional_input_vars": [("u_td", "scalar", 0.0)],
    "sim_code": f"""
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

input_model = {
    "class_name": "input",
    "param_names": ["pop_size", "change_th"],
    "var_name_types": [("r", "scalar"), ("idx_dat", "int"),
                       ("r_last", "scalar"),# ("r_sec_last", "scalar"),
                       ("t_last_spike", "scalar")],
    "sim_code": """
    if($(idx_dat) < $(size_t_sign)){
        if($(t)>=$(t_sign)[$(idx_dat)]*DT){
            $(r) = $(u)[$(id)+$(idx_dat)*$(size_u)];
            $(idx_dat)++;
        }
    }""",
    "threshold_condition_code": TH_COND_CODE,
    "reset_code": RESET_CODE,
    "extra_global_params": [("u", "scalar*"),
                            ("size_u", "int"),
                            ("t_sign", "int*"),
                            ("size_t_sign", "int")],
    "is_auto_refractory_required": False
}
