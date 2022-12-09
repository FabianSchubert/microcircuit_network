"""
Some helper functions for creating the model definitions.
"""

TH_COND_CODE = "abs($(r)-$(r_last)) >= $(change_th)"
RESET_CODE = """
$(r_last) = $(r);
"""


def act_func(x):
    return f'log(1.0+exp(1.0*({x})))'

