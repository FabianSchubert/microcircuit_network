from network_base.synapse import SynapseDense, SynapseSparseOneToOne


class SynapsePPBasal(SynapseDense):

    def __init__(self, *args, **kwargs):

        super().__init__(*args[:-1], **kwargs)

        self.ps_target_var = "Isyn_vb"


class SynapsePINP(SynapseDense):

    def __init__(self, *args, **kwargs):

        super().__init__(*args[:-1], **kwargs)

        self.ps_target_var = "Isyn_vb"


class SynapsePPApical(SynapseDense):

    def __init__(self, *args, **kwargs):

        super().__init__(*args[:-1], **kwargs)

        self.ps_target_var = "Isyn_va_exc"


class SynapseIP(SynapseDense):

    def __init__(self, *args, **kwargs):

        super().__init__(*args[:-1], **kwargs)


class SynapseIPBack(SynapseSparseOneToOne):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.ps_target_var = "u_td"


class SynapsePI(SynapseDense):

    def __init__(self, *args, **kwargs):

        super().__init__(*args[:-1], **kwargs)

        self.ps_target_var = "Isyn_va_int"

class SynapseEquiprop(SynapseDense):

    def __init__(self, *args, **kwargs):

        super().__inist__(*args, **kwargs)

        self.ps_target_var = "Isyn_regular"

