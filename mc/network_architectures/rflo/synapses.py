from network_base.synapse import SynapseDense

class SynapseHidden(SynapseDense):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class SynapseHiddenIn(SynapseDense):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ps_target_var = "Isyn_input"

class SynapseOutHidden(SynapseDense):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ps_target_var = "Isyn_net"

class SynapseHiddenOut(SynapseDense):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ps_target_var = "Isyn_err_fb"
