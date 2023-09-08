from network_base.synapse import SynapseDense

class GenericSynapse(SynapseDense):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.ps_target_var = "Isyn_regular"
