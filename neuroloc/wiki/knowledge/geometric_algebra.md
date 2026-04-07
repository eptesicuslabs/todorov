# Geometric Algebra for Neural Networks

Source: Echoloc project (eptesicuslabs/echoloc)
GATr paper: "Geometric Algebra Transformer"
ArXiv ID: 2305.18415
Authors: Johann Brehmer, Pim de Haan, Sonke Behrends, Taco Cohen (Qualcomm AI Research)
Published: NeurIPS 2023


## 1. Projective Geometric Algebra G(3,0,1)

The algebra G(3,0,1) is the Clifford algebra of 3D projective space.

Basis vector signature:
    e0^2 = 0    (degenerate / null basis vector)
    e1^2 = 1    (Euclidean basis vector)
    e2^2 = 1    (Euclidean basis vector)
    e3^2 = 1    (Euclidean basis vector)

The null vector e0 encodes the projective (homogeneous) coordinate.
This allows representing both directions and positions in the same algebra.


## 2. 16 Basis Blades, 5 Grades

Multivectors in G(3,0,1) have 16 components organized into 5 grades:

    +-------+------+---------------------------------------------+
    | Grade | Dim  | Basis Blades                                |
    +-------+------+---------------------------------------------+
    |   0   |  1   | 1 (scalar)                                  |
    |   1   |  4   | e0, e1, e2, e3                              |
    |   2   |  6   | e01, e02, e03, e12, e13, e23                |
    |   3   |  4   | e012, e013, e023, e123                      |
    |   4   |  1   | e0123 (pseudoscalar)                        |
    +-------+------+---------------------------------------------+
    | Total | 16   |                                             |
    +-------+------+---------------------------------------------+

Geometric objects as multivectors:
- Scalars (grade 0): scalar quantities
- Vectors (grade 1): points and planes
- Bivectors (grade 2): lines and motors (rigid motions)
- Trivectors (grade 3): points (in dual representation)
- Pseudoscalar (grade 4): oriented volume element

Internal component ordering in GATr implementation:
    [x_scalar, x_0, x_1, x_2, x_3, x_01, x_02, x_03,
     x_12, x_13, x_23, x_012, x_013, x_023, x_123, x_0123]


## 3. Cayley Table (Geometric Product)

The geometric product of basis vectors follows from the signature:

    e_i * e_j = -e_j * e_i   for i != j   (anticommutativity)
    e_0 * e_0 = 0
    e_1 * e_1 = 1
    e_2 * e_2 = 1
    e_3 * e_3 = 1

For higher-grade blades, the product is computed by expanding and
applying the above rules. For example:

    e01 * e01 = e0 * e1 * e0 * e1
              = -e0 * e0 * e1 * e1  (swap e1, e0 with sign change)
              = -0 * 1 = 0

    e12 * e12 = e1 * e2 * e1 * e2
              = -e1 * e1 * e2 * e2
              = -1 * 1 = -1

Key products for common operations:
    e12: generates rotations in the e1-e2 plane
    e01, e02, e03: generate translations
    e12, e13, e23: generate rotations

A motor (rigid body transform) is an even-grade multivector:
    M = a + b*e01 + c*e02 + d*e03 + e*e12 + f*e13 + g*e23 + h*e0123


## 4. GATr Approach: Sparse Einsum for Geometric Product

Computing the full geometric product of two 16-component multivectors
requires up to 16*16 = 256 multiplications. However, many products are
zero (due to e0^2 = 0 and cancellations).

GATr implements the geometric product as a sparse einsum:
- Precompute a sparse table of non-zero products
- The bilinear map GP(x, y) is implemented as:
    output[k] = sum_{i,j : cayley[i,j]=k} sign[i,j] * x[i] * y[j]
- This sparse structure significantly reduces computation
- Implemented as a sparse tensor contraction using einsum notation


## 5. Pin-Equivariant Linear Maps

GATr requires all linear maps to be equivariant under Pin(3,0,1), the
double cover of the Euclidean group E(3).

Any Pin(d,0,1)-equivariant linear map phi: G(d,0,1) -> G(d,0,1) has
a highly constrained form. For G(3,0,1), the equivariant linear map is
parameterized by a small number of learnable coefficients.

The equivariant linear layer decomposes as grade projections:
    phi(x) = sum_k w_k * <x>_k + sum_k v_k * <x * e0123>_k

where:
- <x>_k denotes grade-k projection of multivector x
- w_k are learnable scalar weights (one per grade)
- v_k are learnable scalar weights for the dual projection
- e0123 is the pseudoscalar

Total free parameters: 9 (for G(3,0,1))
    - 5 weights w_k (one per grade, k = 0,1,2,3,4)
    - 4 weights v_k (for grades where dual projection is distinct)

This extreme parameter efficiency is a direct consequence of equivariance:
the symmetry group constrains the linear map to have very few degrees of
freedom per input/output channel pair.

In practice, with C_in input channels and C_out output channels:
    Total params = 9 * C_in * C_out


## 6. Grade-Wise RMS Normalization

Standard layer normalization cannot be used because it would mix grades
and break equivariance. GATr uses grade-wise RMS normalization:

    For each grade k:
        rms_k = sqrt(mean(||<x>_k||^2))
        <x>_k_normalized = <x>_k / (rms_k + epsilon)

This normalizes each grade independently, preserving the grade structure
and equivariance of the multivector representation.


## 7. GATr Attention

GATr attention operates on multivector-valued tokens:
- Q, K, V are obtained by applying equivariant linear maps to input multivectors
- Attention scores: computed from the scalar part of the geometric product Q * ~K
  where ~K is the reverse (grade-involution) of K
- The score is a scalar (grade 0), which is invariant under Pin transformations
- Values are aggregated using standard softmax-weighted sum


## 8. Applications Demonstrated

- N-body modeling
- Wall-shear-stress estimation on arterial meshes
- Robotic motion planning
- Consistently outperforms non-geometric and equivariant baselines


## References

- GATr paper: arxiv 2305.18415
- GATr code: https://github.com/Qualcomm-AI-research/geometric-algebra-transformer
- Echoloc project: eptesicuslabs/echoloc
- Choosing a Geometric Algebra for Equivariant Transformers (AISTATS 2024):
  proceedings.mlr.press/v238/haan24a/haan24a.pdf
