import numpy as np
import numpy.linalg as la
import pytest
from state_prep import (
    normalize_amplitudes,
    state_vector_from_amplitudes,
    householder_unitary_for_state,
    prepare_state_unitary,
    apply_unitary_to_zero,
    global_phase_align,
)

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

def test_householder_is_unitary_2q():
    rng = np.random.default_rng(7)
    for _ in range(5):
        a = rng.normal(size=4) + 1j*rng.normal(size=4)
        psi = normalize_amplitudes(a)
        U = householder_unitary_for_state(psi)
        I = U.conj().T @ U
        assert np.allclose(I, np.eye(4), atol=1e-10)

def test_householder_is_unitary_3q():
    rng = np.random.default_rng(11)
    for _ in range(5):
        a = rng.normal(size=8) + 1j*rng.normal(size=8)
        psi = normalize_amplitudes(a)
        U = householder_unitary_for_state(psi)
        I = U.conj().T @ U
        assert np.allclose(I, np.eye(8), atol=1e-10)

def test_householder_identity_when_psi_is_basis_state_2q():
    psi = np.array([1+0j,0,0,0], dtype=np.complex128)
    U = householder_unitary_for_state(psi)
    assert np.allclose(U, np.eye(4))

def test_prepare_2_qubits_matches_up_to_global_phase():
    rng = np.random.default_rng(123)
    a = rng.normal(size=4) + 1j*rng.normal(size=4)
    psi, U = prepare_state_unitary(a)
    out = apply_unitary_to_zero(U)
    out_aligned, _ = global_phase_align(psi, out)
    assert np.allclose(psi, out_aligned, atol=1e-10)

def test_householder_identity_when_psi_is_basis_state_3q():
    psi = np.zeros(8, dtype=np.complex128)
    psi[0] = 1.0 + 0.0j  # |000>
    U = householder_unitary_for_state(psi)
    assert np.allclose(U, np.eye(8))

def test_prepare_3_qubits_matches_up_to_global_phase():
    rng = np.random.default_rng(202)
    a = rng.normal(size=8) + 1j*rng.normal(size=8)
    psi, U = prepare_state_unitary(a)
    out = apply_unitary_to_zero(U)
    out_aligned, _ = global_phase_align(psi, out)
    assert np.allclose(psi, out_aligned, atol=1e-10)
