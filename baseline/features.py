import numpy as np

MAX_ATOMS = 16


def coulomb_matrix_eigenvalues(atomic_numbers, positions, max_atoms=MAX_ATOMS):
    """
    Sorted Coulomb matrix eigenvalues.
    Provably invariant to rotation, translation, and atom permutation.
    Padded to max_atoms with zeros for fixed-size MLP input.

    M_ii = 0.5 * Z_i^2.4
    M_ij = Z_i * Z_j / |r_i - r_j|
    """
    N = len(atomic_numbers)
    M = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        M[i, i] = 0.5 * float(atomic_numbers[i]) ** 2.4
        for j in range(i + 1, N):
            d = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
            if d > 1e-10:
                v = float(atomic_numbers[i]) * float(atomic_numbers[j]) / d
                M[i, j] = v
                M[j, i] = v

    eigs = np.linalg.eigvalsh(M)
    eigs = np.sort(np.abs(eigs))[::-1].astype(np.float32)

    out = np.zeros(max_atoms, dtype=np.float32)
    out[:N] = eigs[:N]
    return out
