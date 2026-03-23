from __future__ import annotations

import torch
from torch import Tensor

from src.algebra.multivector import Multivector, NUM_COMPONENTS, GRADE_RANGES


def _build_cayley_table() -> tuple[Tensor, Tensor]:
    basis_vectors = ["s", "e0", "e1", "e2", "e3"]
    metric = {"e1": 1, "e2": 1, "e3": 1, "e0": 0}

    basis_blades = [
        (),
        (0,), (1,), (2,), (3,),
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3),
        (0, 1, 2, 3),
    ]

    blade_to_index = {}
    for idx, blade in enumerate(basis_blades):
        blade_to_index[blade] = idx

    def multiply_blades(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
        combined = list(a) + list(b)
        sign = 1

        n = len(combined)
        for i in range(n):
            for j in range(i + 1, n):
                if combined[j] < combined[i]:
                    combined[i], combined[j] = combined[j], combined[i]
                    sign *= -1

        result = []
        i = 0
        while i < len(combined):
            if i + 1 < len(combined) and combined[i] == combined[i + 1]:
                vec_idx = combined[i]
                vec_name = basis_vectors[vec_idx + 1] if vec_idx < len(basis_vectors) - 1 else basis_vectors[vec_idx]
                sq = metric.get(basis_vectors[vec_idx + 1], 0) if vec_idx < 4 else 0
                if vec_idx == 0:
                    sq = metric["e0"]
                elif vec_idx == 1:
                    sq = metric["e1"]
                elif vec_idx == 2:
                    sq = metric["e2"]
                elif vec_idx == 3:
                    sq = metric["e3"]
                if sq == 0:
                    return 0, ()
                sign *= sq
                i += 2
            else:
                result.append(combined[i])
                i += 1

        return sign, tuple(result)

    indices = torch.zeros(NUM_COMPONENTS, NUM_COMPONENTS, dtype=torch.long)
    signs = torch.zeros(NUM_COMPONENTS, NUM_COMPONENTS)

    for i, blade_a in enumerate(basis_blades):
        for j, blade_b in enumerate(basis_blades):
            s, result_blade = multiply_blades(blade_a, blade_b)
            if s == 0:
                indices[i, j] = 0
                signs[i, j] = 0.0
            else:
                idx = blade_to_index.get(result_blade, -1)
                if idx == -1:
                    indices[i, j] = 0
                    signs[i, j] = 0.0
                else:
                    indices[i, j] = idx
                    signs[i, j] = float(s)

    return indices, signs


_CAYLEY_INDICES, _CAYLEY_SIGNS = _build_cayley_table()


def _component_grade(index: int) -> int:
    for k, (start, end) in enumerate(GRADE_RANGES):
        if start <= index < end:
            return k
    return -1


def _build_sparse_gp_tables(
    cayley_idx: Tensor, cayley_sgn: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    src_i_list = []
    src_j_list = []
    tgt_k_list = []
    sgn_list = []
    for i in range(NUM_COMPONENTS):
        for j in range(NUM_COMPONENTS):
            if cayley_sgn[i, j] != 0:
                src_i_list.append(i)
                src_j_list.append(j)
                tgt_k_list.append(cayley_idx[i, j].item())
                sgn_list.append(cayley_sgn[i, j].item())
    return (
        torch.tensor(src_i_list, dtype=torch.long),
        torch.tensor(src_j_list, dtype=torch.long),
        torch.tensor(tgt_k_list, dtype=torch.long),
        torch.tensor(sgn_list, dtype=torch.float),
    )


def _build_sparse_outer_tables(
    cayley_idx: Tensor, cayley_sgn: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    src_i_list = []
    src_j_list = []
    tgt_k_list = []
    sgn_list = []
    for i in range(NUM_COMPONENTS):
        grade_i = _component_grade(i)
        for j in range(NUM_COMPONENTS):
            grade_j = _component_grade(j)
            if cayley_sgn[i, j] != 0:
                target = cayley_idx[i, j].item()
                target_grade = _component_grade(target)
                if target_grade == grade_i + grade_j:
                    src_i_list.append(i)
                    src_j_list.append(j)
                    tgt_k_list.append(target)
                    sgn_list.append(cayley_sgn[i, j].item())
    return (
        torch.tensor(src_i_list, dtype=torch.long),
        torch.tensor(src_j_list, dtype=torch.long),
        torch.tensor(tgt_k_list, dtype=torch.long),
        torch.tensor(sgn_list, dtype=torch.float),
    )


def _build_sparse_inner_tables(
    cayley_idx: Tensor, cayley_sgn: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    src_i_list = []
    src_j_list = []
    tgt_k_list = []
    sgn_list = []
    for i in range(NUM_COMPONENTS):
        grade_i = _component_grade(i)
        for j in range(NUM_COMPONENTS):
            grade_j = _component_grade(j)
            if cayley_sgn[i, j] != 0 and grade_i > 0 and grade_j > 0:
                target = cayley_idx[i, j].item()
                target_grade = _component_grade(target)
                if target_grade == abs(grade_i - grade_j):
                    src_i_list.append(i)
                    src_j_list.append(j)
                    tgt_k_list.append(target)
                    sgn_list.append(cayley_sgn[i, j].item())
    return (
        torch.tensor(src_i_list, dtype=torch.long),
        torch.tensor(src_j_list, dtype=torch.long),
        torch.tensor(tgt_k_list, dtype=torch.long),
        torch.tensor(sgn_list, dtype=torch.float),
    )


_GP_SRC_I, _GP_SRC_J, _GP_TGT_K, _GP_SIGNS = _build_sparse_gp_tables(
    _CAYLEY_INDICES, _CAYLEY_SIGNS
)
_OP_SRC_I, _OP_SRC_J, _OP_TGT_K, _OP_SIGNS = _build_sparse_outer_tables(
    _CAYLEY_INDICES, _CAYLEY_SIGNS
)
_IP_SRC_I, _IP_SRC_J, _IP_TGT_K, _IP_SIGNS = _build_sparse_inner_tables(
    _CAYLEY_INDICES, _CAYLEY_SIGNS
)


def _ensure_device(
    device: torch.device,
    src_i: Tensor, src_j: Tensor, tgt_k: Tensor, signs: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if src_i.device != device:
        return src_i.to(device), src_j.to(device), tgt_k.to(device), signs.to(device)
    return src_i, src_j, tgt_k, signs


def _get_cayley_tables(device: torch.device) -> tuple[Tensor, Tensor]:
    global _CAYLEY_INDICES, _CAYLEY_SIGNS
    if _CAYLEY_INDICES.device != device:
        _CAYLEY_INDICES = _CAYLEY_INDICES.to(device)
        _CAYLEY_SIGNS = _CAYLEY_SIGNS.to(device)
    return _CAYLEY_INDICES, _CAYLEY_SIGNS


def _sparse_product(
    a: Multivector, b: Multivector,
    src_i: Tensor, src_j: Tensor, tgt_k: Tensor, signs: Tensor,
) -> Multivector:
    si, sj, tk, sg = _ensure_device(a.device, src_i, src_j, tgt_k, signs)
    sg = sg.to(dtype=a.dtype)
    a_vals = a.values
    b_vals = b.values
    batch_shape = torch.broadcast_shapes(a_vals.shape[:-1], b_vals.shape[:-1])
    a_vals = a_vals.expand(*batch_shape, NUM_COMPONENTS)
    b_vals = b_vals.expand(*batch_shape, NUM_COMPONENTS)
    products = a_vals[..., si] * b_vals[..., sj] * sg
    flat_products = products.reshape(-1, products.shape[-1])
    flat_result = torch.zeros(flat_products.shape[0], NUM_COMPONENTS, device=a.device, dtype=a.dtype)
    flat_result.index_add_(-1, tk, flat_products)
    result = flat_result.reshape(*batch_shape, NUM_COMPONENTS)
    return Multivector(result)


def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    global _GP_SRC_I, _GP_SRC_J, _GP_TGT_K, _GP_SIGNS
    _GP_SRC_I, _GP_SRC_J, _GP_TGT_K, _GP_SIGNS = _ensure_device(
        a.device, _GP_SRC_I, _GP_SRC_J, _GP_TGT_K, _GP_SIGNS
    )
    return _sparse_product(a, b, _GP_SRC_I, _GP_SRC_J, _GP_TGT_K, _GP_SIGNS)


def outer_product(a: Multivector, b: Multivector) -> Multivector:
    global _OP_SRC_I, _OP_SRC_J, _OP_TGT_K, _OP_SIGNS
    _OP_SRC_I, _OP_SRC_J, _OP_TGT_K, _OP_SIGNS = _ensure_device(
        a.device, _OP_SRC_I, _OP_SRC_J, _OP_TGT_K, _OP_SIGNS
    )
    return _sparse_product(a, b, _OP_SRC_I, _OP_SRC_J, _OP_TGT_K, _OP_SIGNS)


def inner_product(a: Multivector, b: Multivector) -> Multivector:
    global _IP_SRC_I, _IP_SRC_J, _IP_TGT_K, _IP_SIGNS
    _IP_SRC_I, _IP_SRC_J, _IP_TGT_K, _IP_SIGNS = _ensure_device(
        a.device, _IP_SRC_I, _IP_SRC_J, _IP_TGT_K, _IP_SIGNS
    )
    return _sparse_product(a, b, _IP_SRC_I, _IP_SRC_J, _IP_TGT_K, _IP_SIGNS)


def sandwich_product(rotor: Multivector, x: Multivector) -> Multivector:
    return geometric_product(geometric_product(rotor, x), rotor.reverse())
