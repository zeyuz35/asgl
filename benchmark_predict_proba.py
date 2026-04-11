import numpy as np
import time

def vstack_predict_proba(decision):
    proba_pos_class = 1 / (1 + np.exp(-decision))
    return np.vstack([1 - proba_pos_class, proba_pos_class]).T

def empty_predict_proba(decision):
    proba_pos_class = 1 / (1 + np.exp(-decision))
    out = np.empty((decision.shape[0], 2), dtype=decision.dtype)
    out[:, 0] = 1 - proba_pos_class
    out[:, 1] = proba_pos_class
    return out

decision = np.random.randn(1000000)

start = time.time()
for _ in range(100):
    vstack_predict_proba(decision)
print("vstack:", time.time() - start)

start = time.time()
for _ in range(100):
    empty_predict_proba(decision)
print("empty:", time.time() - start)

np.testing.assert_array_almost_equal(vstack_predict_proba(decision), empty_predict_proba(decision))
