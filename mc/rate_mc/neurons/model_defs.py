#! /usr/bin/env python3

from ..utils import act_func

pyr_model = {
    "class_name": "pyr",
    "param_names": ["glk", "gb", "ga"],
    "var_name_types": [("u","scalar"), ("r", "scalar"),
                        ("va_int","scalar"),("va_exc","scalar"),
                        ("va","scalar"),
                        ("vb","scalar")],
    "additional_input_vars": [("Isyn_va_int","scalar",0.0),
                            ("Isyn_va_exc","scalar",0.0),
                            ("Isyn_vb","scalar",0.0)],
    "sim_code": f"""
                $(vb) = $(Isyn_vb);
                $(va_int) = $(Isyn_va_int);
                $(va_exc) = $(Isyn_va_exc);
                $(va) = $(va_int) + $(va_exc);
                $(u) += DT * ( -($(glk)+$(gb)+$(ga))*$(u)
                + $(gb)*$(vb)
                + $(ga)*$(va));
                $(r) = {act_func('$(u)')};
                //$(r) = $(u);
                """,
    "threshold_condition_code": None,
    "reset_code": None
}

output_model = {
    "class_name": "output",
    "param_names": ["glk","gb","ga"],
    "var_name_types": [("u","scalar"),("r","scalar"),
    ("vb","scalar"),("gnudge","scalar"),("vnudge","scalar")],
    "additional_input_vars": [("Isyn_vb","scalar",0.0)],
    "sim_code": f"""
                $(vb) = $(Isyn_vb);
                $(u) += DT * (-($(glk)+$(gb)+$(gnudge))*$(u)
                + $(gb)*$(vb)
                + $(gnudge)*$(vnudge));
                $(r) = {act_func('$(u)')};
                """,
    "threshold_condition_code": None,
    "reset_code": None
}  


int_model = {
    "class_name": "int",
    "param_names": ["glk","gd","gsom"],
    "var_name_types": [("u","scalar"),("v","scalar"),("r","scalar")],
    "additional_input_vars": [("u_td","scalar",0.0)],
    "sim_code": f"""
                $(v) = $(Isyn);
                $(u) += DT * ( -$(glk)*$(u)
                + $(gd)*( $(v)-$(u) )
                + $(gsom)*( $(u_td) - $(u) ));// + $(gennrand_normal)*sqrt(DT)*0.1;
                $(r) = {act_func('$(u)')};
                """,
    "threshold_condition_code": None,
    "reset_code": None
}

input_model = {
    "class_name": "input",
    "param_names": None,
    "var_name_types": [("r","scalar"),("u","scalar")],
    "sim_code": "$(r) = $(u);",
    "threshold_condition_code": None,
    "reset_code": None
}