from typing import Iterable, Tuple
import numpy as np


def _is_power_of_two(k: int) -> bool:
    """Return True if k is a power of two (k = 2**m)."""
    return k > 0 and (k & (k - 1)) == 0

def normalize_amplitudes(amplitudes: Iterable[complex]) -> np.ndarray:
    """Return a normalized complex vector as 1D numpy array."""
    a = np.asarray(list(amplitudes), dtype=np.complex128)
    if a.size == 0:
        raise ValueError("Empty amplitude list.")
    norm = np.linalg.norm(a)
    if norm == 0:
        raise ValueError("All amplitudes are zero, normalization undefined.")
    return a / norm

def state_vector_from_amplitudes(amplitudes: Iterable[complex]) -> np.ndarray:
    """Validate length is 2**n for n in {2,3} and return normalized vector."""
    a = np.asarray(list(amplitudes), dtype=np.complex128)
    if not _is_power_of_two(a.size):
        raise ValueError(f"Length {a.size} is not a power of two (2**n).")
    if a.size not in (4, 8):
        raise ValueError(f"Expected length 4 (2 qubits) or 8 (3 qubits), got {a.size}.")
    return normalize_amplitudes(a)

def householder_unitary_for_state(psi: np.ndarray) -> np.ndarray:
    """Construct a unitary U such that U @ e1 = psi using a complex Householder reflection."""
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    d = psi.size
    nrm = np.linalg.norm(psi)
    if not np.isclose(nrm, 1.0, atol=1e-12):
        psi = psi / nrm
    e1 = np.zeros(d, dtype=np.complex128); e1[0] = 1.0 + 0.0j
    if np.allclose(psi, e1, atol=1e-12):
        return np.eye(d, dtype=np.complex128)
    
    if abs(psi[0]) > 0:
        alpha = psi[0] / abs(psi[0])          # unit-modulus complex
    else:
        alpha = 1.0 + 0.0j                    # arbitrary; psi[0]==0 -> no phase info
    psi_prime = np.conjugate(alpha) * psi     # (psi')_0 is now real and >= 0

    v = e1 - psi_prime
    beta = np.vdot(v, v)
    if beta <= 1e-32:
        return alpha * np.eye(d, dtype=np.complex128)
    H = np.eye(d, dtype=np.complex128) - 2.0 * np.outer(v, np.conjugate(v)) / beta
    U = alpha * H
    return U

def prepare_state_unitary(amplitudes: Iterable[complex]) -> Tuple[np.ndarray, np.ndarray]:
    """From amplitudes -> (psi, U) with U @ |0...0> = psi."""
    psi = state_vector_from_amplitudes(amplitudes)
    U = householder_unitary_for_state(psi)
    return psi, U

def apply_unitary_to_zero(U: np.ndarray) -> np.ndarray:
    """Return U @ e1 (quantum state produced by applying U to |0...0>)."""
    d = U.shape[0]
    e1 = np.zeros(d, dtype=np.complex128); e1[0] = 1.0 + 0.0j
    return U @ e1

def global_phase_align(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, complex]:
    """Align the global phase of y to x and return (y_aligned, phase)."""
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    y = np.asarray(y, dtype=np.complex128).reshape(-1)
    inner = np.vdot(x, y)
    if np.isclose(inner, 0.0, atol=1e-14):
        return y, 1.0 + 0.0j
    phase = np.conj(inner) / abs(inner)
    return y * phase, phase
