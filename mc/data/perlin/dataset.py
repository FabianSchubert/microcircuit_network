import numpy as np
import NPerlinNoise as nPN


def contrast(x, g=1.0):

    a0 = np.power(np.abs(x), g)
    return a0 / (np.power(np.abs(x-1.), g) + a0)


def perlin_dataset_array(n_in=2, size=1000, seed_noise=42, seed_input=None):

    noise = nPN.Noise(seed=seed_noise, frequency=3, waveLength=1, octaves=2,
                      persistence=0.85, lacunarity=2)

    np.random.seed(seed_input)

    x = np.random.rand(size, n_in)
    y = noise(*x.T.tolist())
    y = np.expand_dims(y, axis=-1)

    y = contrast(y, g=2.)
    y = (y-y.min())*0.8/(y.max()-y.min()) + 0.1

    return x, y
