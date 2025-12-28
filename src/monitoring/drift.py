import numpy as np

def psi(expected, actual, buckets=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    psi_value = 0

    for i in range(buckets):
        e = np.mean((expected >= breakpoints[i]) & (expected < breakpoints[i+1]))
        a = np.mean((actual >= breakpoints[i]) & (actual < breakpoints[i+1]))

        if e > 0 and a > 0:
            psi_value += (a - e) * np.log(a / e)

    return psi_value