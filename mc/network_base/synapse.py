from dataclasses import dataclass, field
import typing

from .utils import (merge_wu_def, merge_dicts,
                    merge_ps_def,
                    param_change_batch_reduce,
                    optimizers)

from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY

from pygenn.genn_model import (create_custom_weight_update_class,
                               create_custom_postsynaptic_class,
                               init_connectivity,
                               create_wu_var_ref)

from copy import deepcopy

DEFAULT_OPTIM = {
    "optimizer": "adam",
    "params": {
        "low": -1000., "high": 1000.,
        "lr": 1e-3,
        "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-7
    }
}


@dataclass
class SynapseBase:
    matrix_type: str = None
    delay_steps: int = 0
    w_update_model_transmit: dict = field(default_factory=dict)
    w_update_model_plast: dict = field(default_factory=dict)
    wu_param_space_transmit: dict = field(default_factory=dict)
    wu_param_space_plast: dict = field(default_factory=dict)
    wu_var_space_transmit: dict = field(default_factory=dict)
    wu_var_space_plast: dict = field(default_factory=dict)
    wu_pre_var_space_transmit: dict = field(default_factory=dict)
    wu_pre_var_space_plast: dict = field(default_factory=dict)
    wu_post_var_space_transmit: dict = field(default_factory=dict)
    wu_post_var_space_plast: dict = field(default_factory=dict)
    ps_model_transmit: dict = field(default_factory=dict)
    ps_model_plast: dict = field(default_factory=dict)
    ps_param_space_transmit: dict = field(default_factory=dict)
    ps_param_space_plast: dict = field(default_factory=dict)
    ps_var_space_transmit: dict = field(default_factory=dict)
    ps_var_space_plast: dict = field(default_factory=dict)
    connectivity_initialiser: "typing.Any" = None
    ps_target_var: str = "Isyn"
    norm_after_init: typing.Any = False,
    low: float = -10000.0
    high: float = 10000.0

    def build_wu_model(self, plastic, read_only=True):
        '''
        Build either a "plastic" weight update
        model by merging the transmission and
        plastic weight update definition, or
        a static model using only the transmission
        model definition.
        '''

        if plastic:
            _wu_model_def = merge_wu_def("plastic_wu_model",
                                               self.w_update_model_transmit,
                                               self.w_update_model_plast)

        else:
            _wu_model_def = deepcopy(self.w_update_model_transmit)

        if read_only:
            _var_name_types = _wu_model_def["var_name_types"]
            idx_g = _var_name_types.index(("g", "scalar"))
            _var_name_types[idx_g] = ("g", "scalar", VarAccess_READ_ONLY)

        return create_custom_weight_update_class(**_wu_model_def)

    def build_ps_model(self, plastic):
        '''
        Build either a "plastic" postsynaptic
        model by merging the transmission and
        plastic postsynaptic model definition, or
        a static model using only the transmission
        model definition.
        '''
        if plastic:
            merged_ps_model_def = merge_ps_def("plastic_ps_model",
                                               self.ps_model_transmit,
                                               self.ps_model_plast)

            return create_custom_postsynaptic_class(**merged_ps_model_def)

        return create_custom_postsynaptic_class(**self.ps_model_transmit)

    def connect_pops(self, name, genn_model,
                     target, source, plastic=True, read_only=True,
                     optimizer_params={}):

        wu_model = self.build_wu_model(plastic, read_only)
        ps_model = self.build_ps_model(plastic)

        _optimizer_params = optimizer_params.get(name, DEFAULT_OPTIM)

        if plastic:
            wu_param_space = merge_dicts(self.wu_param_space_plast,
                                         self.wu_param_space_transmit)
            wu_var_space = merge_dicts(self.wu_var_space_plast,
                                       self.wu_var_space_transmit)

            wu_pre_var_space = merge_dicts(self.wu_pre_var_space_plast,
                                           self.wu_pre_var_space_transmit)
            wu_post_var_space = merge_dicts(self.wu_post_var_space_plast,
                                            self.wu_post_var_space_transmit)

            ps_param_space = merge_dicts(self.ps_param_space_plast,
                                         self.ps_param_space_transmit)
            ps_var_space = merge_dicts(self.ps_var_space_plast,
                                       self.ps_var_space_transmit)

        else:
            wu_param_space = dict(self.wu_param_space_transmit)
            wu_var_space = dict(self.wu_var_space_transmit)
            wu_pre_var_space = dict(self.wu_pre_var_space_transmit)
            wu_post_var_space = dict(self.wu_post_var_space_transmit)
            ps_param_space = dict(self.ps_param_space_transmit)
            ps_var_space = dict(self.ps_var_space_transmit)

        _syn_pop = genn_model.add_synapse_population(
            name, self.matrix_type, self.delay_steps,
            source, target, wu_model, wu_param_space,
            wu_var_space, wu_pre_var_space,
            wu_post_var_space, ps_model,
            ps_param_space, ps_var_space,
            self.connectivity_initialiser
        )


        _syn_pop.ps_target_var = self.ps_target_var

        _syn_pop.norm_after_init = self.norm_after_init

        if plastic:

            _update_reduce_batch_weight_change_var_refs = {
                "change": create_wu_var_ref(_syn_pop, "dg")
            }

            _update_reduce_batch_weight_change = genn_model.add_custom_update(
                                         f"reduce_batch_weight_change_{name}",
                                         "WeightChangeBatchReduce",
                                         param_change_batch_reduce,
                                         {}, {"reducedChange": 0.0},
                                         _update_reduce_batch_weight_change_var_refs)



            _update_plast_step_reduced_var_refs = {
                "change": create_wu_var_ref(_update_reduce_batch_weight_change, "reducedChange"),
                "variable": create_wu_var_ref(_syn_pop, "g")
            }

            optimizer = optimizers[_optimizer_params["optimizer"]]

            genn_model.add_custom_update(
                f"plast_step_reduced_{name}",
                "Plast",
                optimizer["model"],
                {"batch_size": genn_model.batch_size} | _optimizer_params["params"],
                optimizer["var_init"],
                _update_plast_step_reduced_var_refs
            )

        return _syn_pop


class SynapseDense(SynapseBase):
    '''
    SynapseDense extends the base synapse
    class by specifying the matrix type
    as "DENSE_INDIVIDUALG".
    '''

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.matrix_type = "DENSE_INDIVIDUALG"


class SynapseSparseOneToOne(SynapseBase):
    '''
    SynapseSparseOneToOne extends the
    base synapse class by specifying the
    matrix type as "SPARSE_INDIVIDUALG", which
    refers to a sparse matrix representation
    and an individual set of variables for
    every synapse in a population. Furthermore,
    the connectivity initialiser is set to be
    "OneToOne", meaning that this synapse class
    can only connect populations of the same size.
    '''

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.matrix_type = "SPARSE_INDIVIDUALG"
        self.connectivity_initialiser = init_connectivity("OneToOne", {})
