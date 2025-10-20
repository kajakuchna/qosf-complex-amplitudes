import numpy as np
import numpy.linalg as la
import pytest
from state_prep import normalize_amplitudes, state_vector_from_amplitudes


def test_normalize_nonzero_vector():
    a = np.array([1+1j, 0, 0, 0], dtype=np.complex128)
    x = normalize_amplitudes(a)
    assert np.isclose(la.norm(x), 1.0)

def test_normalize_raises_on_zero_vector():
    with pytest.raises(ValueError):
        normalize_amplitudes([0,0,0,0])

def test_state_vector_checks_length_power_of_two():
    with pytest.raises(ValueError):
        state_vector_from_amplitudes([1,0,0])  # Length 3 not power of two

def test_state_vector_rejects_lengths_out_of_scope():
    with pytest.raises(ValueError):
        state_vector_from_amplitudes([0]*16)   # Length 16 not in {4,8}