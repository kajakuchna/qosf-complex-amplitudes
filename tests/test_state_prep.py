import numpy as np
import numpy.linalg as la
import pytest
from state_prep import normalize_amplitudes


def test_normalize_nonzero_vector():
    a = np.array([1+1j, 0, 0, 0], dtype=np.complex128)
    x = normalize_amplitudes(a)
    assert np.isclose(la.norm(x), 1.0)

def test_normalize_raises_on_zero_vector():
    with pytest.raises(ValueError):
        normalize_amplitudes([0,0,0,0])
