'''
Args:
    class_name          --  name of the new class
    
    Keyword args:
    param_names         --  list of strings with param names of the model
    var_name_types      --  list of pairs of strings with varible names and
                            types of the model
    derived_params      --  list of pairs, where the first member is string
                            with name of the derived parameter and the second
                            should be a functor returned by create_dpf_class
    decay_code          --  string with the decay code
    apply_input_code    --  string with the apply input code
    support_code        --  string with the support code
    extra_global_params --  list of pairs of strings with names and
                            types of additional parameters
    custom_body         --  dictionary with additional attributes and methods
                            of the new class
'''

integrator_update = {
    "class_name": "integrator_update",
    "apply_input_code": "$(Isyn) += $(inSyn);"
}