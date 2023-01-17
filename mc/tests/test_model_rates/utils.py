"""
Some helper functions for creating the model definitions.
"""

TH_COND_CODE = None
RESET_CODE = None

def act_func(x):
    return f'log(1.0+exp(1.0*({x})))'

