from pygenn.genn_model import create_var_ref
from .utils import optimizers, param_change_batch_reduce

DEFAULT_OPTIM = {
    "optimizer": "adam",
    "params": {
        "low": -1000., "high": 1000.,
        "lr": 1e-3,
        "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-7
    }
}


class LayerBase:

    def __init__(self, name, genn_model, plastic=True, read_only_weights=False):
        self.name = name
        self.genn_model = genn_model
        self.neur_pops = {}
        self.syn_pops = {}
        self.plastic = plastic
        self.read_only_weights = read_only_weights

    def add_neur_pop(self, pop_name, size, neur_model, param_init, var_init, 
                     optimizer_params={}):
        _full_name = f'neur_{self.name}_{pop_name}'
        _new_pop = self.genn_model.add_neuron_population(_full_name, size,
                                                         neur_model,
                                                         param_init,
                                                         var_init)

        _optimizer_params = optimizer_params.get(_full_name, DEFAULT_OPTIM)

        if self.plastic:

            _update_reduce_batch_bias_change_var_refs = {
                "change": create_var_ref(_new_pop, "db")
            }

            _update_reduce_batch_bias_change = self.genn_model.add_custom_update(
                                         f"reduce_batch_bias_change_{_full_name}",
                                         "BiasChangeBatchReduce",
                                         param_change_batch_reduce,
                                         {}, {"reducedChange": 0.0},
                                         _update_reduce_batch_bias_change_var_refs)



            _update_plast_step_reduced_var_refs = {
                "change": create_var_ref(_update_reduce_batch_bias_change, "reducedChange"),
                "variable": create_var_ref(_new_pop, "b")
            }

            optimizer = optimizers[_optimizer_params["optimizer"]]

            self.genn_model.add_custom_update(
                f"plast_step_reduced_{_full_name}",
                "Plast",
                optimizer["model"],
                {"batch_size": self.genn_model.batch_size} | _optimizer_params["params"],
                optimizer["var_init"],
                _update_plast_step_reduced_var_refs
            )

        self.neur_pops[pop_name] = _new_pop

    def add_syn_pop(self, target, source, syn_model, optimizer_params={}):

        self.syn_pops[f'{source}_to_{target}'] = syn_model.connect_pops(
            f'syn_{self.name}_{source}_to_{target}',
            self.genn_model, self.neur_pops[target], self.neur_pops[source],
            plastic=self.plastic, read_only=self.read_only_weights, optimizer_params=optimizer_params)
