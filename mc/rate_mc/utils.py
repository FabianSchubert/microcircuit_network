#! /usr/bin/env python3

act_func = lambda x: f'log(1.0+exp(1.0*({x})))'
#act_func = lambda x: f'tanh({x})'
#act_func = lambda x: f'{x}'

def merge_dicts(dict_1,dict_2):

	return dict_1 | dict_2

	'''
	dict_merge = dict(dict_1)

	for k,v in dict_2.items():
		if k not in dict_merge.keys():
			dict_merge[k] = v
		else:
			assert dict_merge[k] == v, "conflicting values for mulitple occurences of the same key"

	return dict_merge
	'''


def merge_dict_list_strings(dict_1,dict_2,key):

	assert all(isinstance(s,str) for s in dict_1.get(key,[])), f"{key} list 1 contains non-string elements"
	assert all(isinstance(s,str) for s in dict_2.get(key,[])), f"{key} list 2 contains non-string elements"

	return list(set(dict_1.get(key,[])+dict_2.get(key,[])))


def merge_dict_tuple_args(dict_1,dict_2,key):

	dict_list_1 = dict(dict_1.get(key,[]))
	dict_list_2 = dict(dict_2.get(key,[]))

	assert all(isinstance(s,str) for s in dict_list_1.keys()), "List 1 contains non-string names"
	assert all(isinstance(s,str) for s in dict_list_2.keys()), "List 2 contains non-string names"

	dict_merge = merge_dicts(dict_list_1,dict_list_2)

	return list(dict_merge.items())


def merge_wu_def(class_name,def_1,def_2):

	wu_def = {
	"class_name": class_name
	}

	wu_def["param_names"] = merge_dict_list_strings(def_1,def_2,"param_names")

	var_params_pair_names = ["var_name_types",
						  	"pre_var_name_types",
						  	"post_var_name_types",
						  	"derived_params",
						  	"extra_global_params"]

	for vp in var_params_pair_names:
		wu_def[vp] = merge_dict_tuple_args(def_1,def_2,vp)

	code_strings = ["sim_code",
		  	"event_code",
		  	"learn_post_code",
		  	"synapse_dynamics_code",
		  	"event_threshold_condition_code",
		  	"pre_spike_code",
		  	"post_spike_code",
		  	"pre_dynamics_code",
		  	"post_dynamics_code",
		  	"sim_support_code",
		  	"learn_post_support_code",
		  	"synapse_dynamics_suppport_code"]

	for code in code_strings:
		code_1 = def_1.get(code,None)
		code_2 = def_2.get(code,None)
		if(code_1 is None and code_2 is None):
			wu_def[code] = None
		else:
			code_1 = "" if code_1 == None else code_1
			code_2 = "" if code_2 == None else code_2

			wu_def[code] = f'''{code_1}\n{code_2}'''

	### check additional boolean requirements parameters

	boolean_flags = ["is_pre_spike_time_required",
					"is_post_spike_time_required",
				  	"is_pre_spike_event_time_required",
				  	"is_prev_pre_spike_time_required",
				  	"is_prev_post_spike_time_required",
				  	"is_prev_pre_spike_event_time_required"]

	for flag in boolean_flags:
		wu_def[flag] = (def_1.get(flag,None) is True) or (def_2.get(flag,None) is True)

	return wu_def