def act_func(x):
    #return f'max(0.0, min(1.0, {x}))'
    return f'((tanh(2.0*({x}))+1.0)/2.0)'


def d_act_func(x):
    #return f'(((({x}) > 1.0 ) || (({x}) < 0.0 )) ? 0.0 : 1.0)'
    return f'(1.0-tanh(2.0*({x}))*tanh(2.0*({x})))'
