def act_func(x):
    return f'max(0.0, min(1.0, {x}))'


def d_act_func(x):
    return f'(((({x}) > 1.0 ) || (({x}) < 0.0 )) ? 0.0 : 1.0)'
