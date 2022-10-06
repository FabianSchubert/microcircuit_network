#! /usr/bin/env python3

from .weight_update.models import (wu_model_pp_basal, 
                                            wu_model_pp_apical,
                                            wu_model_pi,
                                            wu_model_ip,
                                            wu_model_pi_back,
                                            wu_model_inpp)
from .weight_update.params import (wu_param_space_pp_basal, 
                                            wu_param_space_pp_apical,
                                            wu_param_space_pi,
                                            wu_param_space_ip,
                                            wu_param_space_pi_back,
                                            wu_param_space_inpp)
from .weight_update.var_inits import (wu_var_space_pp_basal,
                                            wu_var_space_pp_apical,
                                            wu_var_space_pi,
                                            wu_var_space_ip,
                                            wu_var_space_inpp)

from pygenn.genn_model import init_var

from pygenn.genn_model import GeNNModel, init_connectivity

from dataclasses import dataclass, field

@dataclass
class SynapseBase:
    matrix_type: str = None# = "DENSE_INDIVIDUALG"
    delay_steps: int = 0
    w_update_model:"typing.Any" = None
    wu_param_space: dict = field(default_factory=dict)
    wu_var_space: dict = field(default_factory=dict)
    wu_pre_var_space: dict = field(default_factory=dict)
    wu_post_var_space: dict = field(default_factory=dict)
    postsyn_model: str = "DeltaCurr"
    ps_param_space: dict = field(default_factory=dict)
    ps_var_space: dict = field(default_factory=dict)
    connectivity_initialiser: "typing.Any" = None
    ps_target_var: str = "Isyn"
    


    def connect_pops(self,name,genn_model,source,target):

        _syn_pop = genn_model.add_synapse_population(
            name,self.matrix_type,self.delay_steps,
            source,target,self.w_update_model,self.wu_param_space,
            self.wu_var_space,self.wu_pre_var_space,
            self.wu_post_var_space,self.postsyn_model,
            self.ps_param_space,self.ps_var_space,
            self.connectivity_initialiser
        )
        _syn_pop.ps_target_var = self.ps_target_var
        
        return _syn_pop

class SynapseDense(SynapseBase):

    def __init__(self,*args,**kwargs):

        super().__init__(*args,**kwargs)

        self.matrix_type = "DENSE_INDIVIDUALG"


class SynapseSparseOneToOne(SynapseBase):

    def __init__(self,*args,**kwargs):

        super().__init__(*args,**kwargs)

        self.matrix_type = "SPARSE_GLOBALG"
        self.connectivity_initialiser = init_connectivity("OneToOne",{})


class SynapsePPBasal(SynapseDense):
    
    def __init__(self,*args,**kwargs):

        super().__init__(*args[:-1],**kwargs)

        N_norm = args[-1]

        self.w_update_model = wu_model_pp_basal
        self.wu_param_space = wu_param_space_pp_basal
        self.wu_var_space = {
            "g": init_var("Uniform", 
                {"min": -1./N_norm**.5, "max": 1./N_norm**.5}),
                "vbEff": 0.0,
                "dg": 0.0
            }
        #self.wu_var_space = wu_var_space_pp_basal
        self.ps_target_var = "Isyn_vb"

class SynapseINPP(SynapseDense):

    def __init__(self,*args,**kwargs):

        super().__init__(*args[:-1],**kwargs)

        N_norm = args[-1]

        self.w_update_model = wu_model_inpp
        self.wu_param_space = wu_param_space_inpp
        self.wu_var_space = {
            "g": init_var("Uniform", 
                {"min": -1./N_norm**.5, "max": 1./N_norm**.5}),
                "vbEff": 0.0,
                "dg": 0.0 
        }
        #self.wu_var_space = wu_var_space_pp_basal
        self.ps_target_var = "Isyn_vb"

class SynapsePPApical(SynapseDense):
    
    def __init__(self,*args,**kwargs):

        super().__init__(*args[:-1],**kwargs)

        N_norm = args[-1]

        self.w_update_model = wu_model_pp_apical
        self.wu_param_space = wu_param_space_pp_apical
        self.wu_var_space = {
            "g": init_var("Uniform", 
                {"min": -1./N_norm**.5, "max": 1./N_norm**.5})
        }
        #self.wu_var_space = wu_var_space_pp_apical
        self.ps_target_var = "Isyn_va_exc"

class SynapsePI(SynapseDense):
    
    def __init__(self,*args,**kwargs):

        super().__init__(*args[:-1],**kwargs)

        N_norm = args[-1]

        self.w_update_model = wu_model_pi
        self.wu_param_space = wu_param_space_pi
        #self.wu_var_space = wu_var_space_pi
        self.wu_var_space = {
            "g": init_var("Uniform", 
                {"min": -1./N_norm**.5, "max": 1./N_norm**.5}),
                "vEff": 0.0,
                "dg": 0.0 
        }
        #self.ps_target_var = "Isyn"

class SynapsePIBack(SynapseSparseOneToOne):

    def __init__(self,*args,**kwargs):

        super().__init__(*args,**kwargs)

        self.w_update_model = wu_model_pi_back
        self.wu_param_space = wu_param_space_pi_back
        self.wu_var_space = {
            "g": 1.0
        }
        self.ps_target_var = "u_td"

class SynapseIP(SynapseDense):
    
    def __init__(self,*args,**kwargs):

        super().__init__(*args[:-1],**kwargs)

        N_norm = args[-1]

        self.w_update_model = wu_model_ip
        self.wu_param_space = wu_param_space_ip
        #self.wu_var_space = wu_var_space_ip
        self.wu_var_space = {
            "g": init_var("Uniform", 
                {"min": -1./N_norm**.5, "max": 1./N_norm**.5}),
            "dg": 0.0
        }
        self.ps_target_var = "Isyn_va_int"


#synapse_pp_basal = SynapsePPBasal()
#synapse_pp_apical = SynapsePPApical()
#synapse_pi = SynapsePI()
#synapse_ip = SynapseIP()


'''
# Entries set to None should be set individually,
synapse_dense = {
    "w_update_model": None,
    "matrix_type": "DENSE_INDIVIDUALG",
    "delay_steps": 0,
    "wu_param_space": None,
    "wu_var_space": None,
    "wu_pre_var_space": None,
    "w_post_var_space": None,
    "postsyn_model": "DeltaCurr",
    "ps_param_space": {},
    "ps_var_space": {},
    "connectivity_initialiser": None
}

# Pyr to pyr Basal Fwd
synapse_pp_basal = dict(synapse_dense)
synapse_pp_basal["w_update_model"] = wu_model_pp_basal
synapse_pp_basal["wu_param_space"] = wu_param_space_pp_basal
synapse_pp_basal["wu_var_space"] = wu_var_space_pp_basal
synapse_pp_basal["ps_target_var"] = "Isyn_vb"

# Pyr to pyr Apical Back
synapse_pp_apical = dict(synapse_dense)
synapse_pp_apical["w_update_model"] = wu_model_pp_apical
synapse_pp_apical["wu_param_space"] = wu_param_space_pp_apical
synapse_pp_apical["wu_var_space"] = wu_var_space_pp_apical
synapse_pp_apical["ps_target_var"] = "Isyn_va_exc"

# Pyr to int within layer
synapse_pi = dict(synapse_dense)
synapse_pi["w_update_model"] = wu_model_pi
synapse_pi["wu_param_space"] = wu_param_space_pi
synapse_pi["wu_var_space"] = wu_var_space_pi

# Int to pyr within layer
synapse_ip = dict(synapse_dense)
synapse_ip["w_update_model"] = wu_model_ip
synapse_ip["wu_param_space"] = wu_param_space_ip
synapse_ip["wu_var_space"] = wu_var_space_ip
synapse_ip["ps_target_var"] = "Isyn_va_int"

'''