import numpy as np

def generate_ordered(rs, n):
    # generate list of ordered random numbers in linear time
    samples = -1 * np.log(rs.uniform(size=n))
    G = samples.sum() - np.log(rs.uniform())
    samples /= G
    return samples.cumsum()

def sample_from_cdf(rs, cdf, n_samples):
    samples = []
    i = 0
    randomness = generate_ordered(rs, n_samples)
    for r in randomness:
        while r > cdf[i]:
            i += 1
        samples.append(i)
    return samples