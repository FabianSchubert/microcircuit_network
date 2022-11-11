## Stability Issues

After approx. 300000 time steps, the simulation
becomes unstable and dynamic variables start to
diverge. Even before the simulation brakes down,
transient oscillations can be observed that get
more prominent when approaching the critical
point in time.

## Tracking the Neural Input Current

The idea of this implementation is to
temporally coarse-grain the propagation of
neural activity through the network. To keep
track of the correct postsynaptic input current,
each "spike event" would cause the postsynaptic
current to be updated via

```
Isyn += g * (r_pre - r_last_pre)
```
where g is the synaptic weight, r_pre the current
presynaptic rate and r_last_pre the presynaptic
rate at the last update. If Isyn was initialized to
its exact value in the beginning (Iysin = 0) and
g only changed very slowly, this update would correctly
update postsynaptic current whenever an event
is triggered. HOWEVER, it appears to be the case that
with the chosen learning rates, the changes in g in
between update steps is enough quickly induce errors
in tracking the correct input current.

Setting the learning rates to zero will lead to
perfect tracking of the Input currents. It seems that
a different approach to tracking changes in the input
current is required.

### Idea:

Instead of tracking previous rates in the neuron
model (which is more memory efficient), one could implement
a weight update model that contains additional variables
storing the precise contribution to the input current
in the previous time step, i.e. ``g(t_last) * r_pre(t_last)``.
Obviously, this would increase memory quadratically, as
n_pre * n_post of these additional variables need to be stored
(in the dense matrix case).
