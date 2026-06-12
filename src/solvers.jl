# solvers.jl - Unified batched minimum-norm solver

# ------------------------------- Backend Types -------------------------------

abstract type SolverBackend end

"""
MAGMA-based Cholesky solver for minimum-norm solutions.
Uses batched Cholesky factorization via MAGMA library.
Set `preconditioner` to `:diagonal` for improved stability.
"""
struct MAGMACholesky <: SolverBackend
    preconditioner::Symbol  # :none, :diagonal
end
MAGMACholesky(; preconditioner::Symbol=:none) = MAGMACholesky(preconditioner)

"""
cuSOLVER-based Cholesky solver for minimum-norm solutions.
Native CUDA solution without external dependencies.
Set `preconditioner` to `:diagonal` for improved stability.
"""
struct cuSOLVERCholesky <: SolverBackend
    preconditioner::Symbol  # :none, :diagonal
end
cuSOLVERCholesky(; preconditioner::Symbol=:none) = cuSOLVERCholesky(preconditioner)

"""
MAGMA-based QR solver for nullspace projection.
Uses batched QR factorization of Gᵀ via MAGMA, avoiding formation of GGᵀ.
Numerically more stable than Cholesky-based approaches.
"""
struct MAGMAQR <: SolverBackend end

"""
cuSOLVER/CUBLAS-based QR solver for nullspace projection.
Uses batched QR factorization of Gᵀ via CUBLAS, avoiding formation of GGᵀ.
Numerically more stable than Cholesky-based approaches.
"""
struct cuSOLVERQR <: SolverBackend end

# -------------------------------- Cache Type --------------------------------

"""
Cache for minimum-norm solver backward pass (Cholesky-based).
Stores Cholesky factor L, intermediate solution y, and final solution x.
"""
struct MinNormCache
    L::CuArray{Float32,3}
    y::CuArray{Float32,3}
    x::CuArray{Float32,3}
end

"""
Cache for minimum-norm solver backward pass (QR-based).
Stores the R factor from QR of Aᵀ, intermediate solution y, and final solution x.
"""
struct MinNormCacheQR
    R::CuArray{Float32,3}
    y::CuArray{Float32,3}
    x::CuArray{Float32,3}
end


# ------------------------------ Function Stubs ------------------------------

"""
    min_norm_forward(A, b, backend)

Forward pass for batched minimum-norm solution.
Returns (x, cache) where x = A⁺b and cache is used for backward pass.
"""
function min_norm_forward end

"""
    min_norm_backward(A, cache, grad_x, backend)

Backward pass for batched minimum-norm solution.
Returns (grad_A, grad_b).
"""
function min_norm_backward end

# ------------------------------ GPU Public API ------------------------------

"""
    min_norm_solve(A, b, backend)

Batched minimum-norm solution x = A⁺b.

Dispatches to the appropriate solver based on `backend`:
- `MAGMACholesky()`: Uses MAGMA Cholesky factorization - see magma.jl
- `cuSOLVERCholesky()`: Uses native cuSOLVER - see cusolver.jl
- `MAGMAQR()`: Uses MAGMA QR factorization - see magma.jl
- `cuSOLVERQR()`: Uses CUBLAS QR factorization - see cusolver.jl

# Arguments
- `A`: Matrix, size (m, n, batch) where m < n
- `b`: Right-hand side, size (m, batch)
- `backend`: Solver backend

# Returns
- Minimum-norm solution x, size (n, batch)
"""
function min_norm_solve(
    A::CuArray{Float32,3},
    b::CuArray{Float32,2},
    backend::SolverBackend;
)
    return first(min_norm_forward(A, b, backend))
end

function ChainRulesCore.rrule(
    ::typeof(min_norm_solve),
    A::CuArray{Float32,3},
    b::CuArray{Float32,2},
    backend::SolverBackend;
)
    x, cache = min_norm_forward(A, b, backend)
    function pullback(grad_x)
        grad_A, grad_b = min_norm_backward(A, cache, grad_x, backend)
        (NoTangent(), grad_A, grad_b, NoTangent())
    end
    return x, pullback
end

# ------------------------------- CPU Fallback -------------------------------

# CPU min_norm_solve with backend argument (forwards to no-backend version)
function min_norm_solve(
    A::AbstractArray{T,3},
    b::AbstractArray{T,2},
    ::SolverBackend,
) where {T}
    return min_norm_solve(A, b)
end

# CPU min_norm_solve using Julia's native linear algebra
# Uses reduce/hcat instead of mutation for Zygote compatibility
function min_norm_solve(
    A::AbstractArray{T,3},
    b::AbstractArray{T,2},
) where {T}
    m, n, batch = size(A)
    slices = map(1:batch) do i
        Ai = A[:, :, i]
        bi = b[:, i]
        # x = Aᵀ(AAᵀ)⁻¹b
        AAt = Ai * Ai'
        Ai' * (AAt \ bi)
    end
    return reduce(hcat, slices)
end

# ---------------------- Dispatch Wrappers ----------------------

spotrf_batched!(A, ::MAGMACholesky) = magma_spotrf_batched!(A)
spotrf_batched!(A, ::cuSOLVERCholesky) = cusolver_spotrf_batched!(A)

spotrs_batched!(L, B, ::MAGMACholesky) = magma_spotrs_batched!(L, B)
spotrs_batched!(L, B, ::cuSOLVERCholesky) = cusolver_spotrs_batched!(L, B)

# ---------------------- GPU Min-Norm Solver ----------------------

# MAGMA Cholesky solve
function min_norm_forward(A::CuArray{Float32,3}, b::CuArray{Float32,2}, ::MAGMACholesky)
    m, n, batch_size = size(A)

    # AAᵀ: (m, m, batch)
    At = NNlib.batched_transpose(A)  # (n, m, batch)
    L = NNlib.batched_mul(A, At)     # (m, m, batch)

    # Combined Cholesky factorization + solve: L overwritten with factor, y with solution
    y = reshape(copy(b), m, 1, batch_size)
    magma_sposv_batched!(L, y)

    # x = Aᵀy
    x = NNlib.batched_mul(At, y)  # (n, 1, batch)

    return dropdims(x; dims = 2), MinNormCache(L, y, x)
end

# Generic Cholesky implementation for other backends
# Combined Cholesky factorization + solve
function min_norm_forward(A::CuArray{Float32,3}, b::CuArray{Float32,2}, backend::SolverBackend)
    m, n, batch_size = size(A)

    # AAᵀ: (m, m, batch)
    At = NNlib.batched_transpose(A)  # (n, m, batch)
    L = NNlib.batched_mul(A, At)     # (m, m, batch)

    # Cholesky: L where AAᵀ = LLᵀ (in-place)
    spotrf_batched!(L, backend)

    # Solve LLᵀy = b
    y = reshape(copy(b), m, 1, batch_size)
    spotrs_batched!(L, y, backend)

    # x = Aᵀy
    x = NNlib.batched_mul(At, y)  # (n, 1, batch)

    return dropdims(x; dims = 2), MinNormCache(L, y, x)
end

function min_norm_backward(
    A::CuArray{Float32,3},
    cache::MinNormCache,
    grad_x::CuArray{Float32,2},
    backend::SolverBackend,
)
    m, n, batch_size = size(A)
    (; L, y, x) = cache
    grad_x_3d = reshape(grad_x, n, 1, batch_size)

    # z = (AAᵀ)⁻¹(A·grad_x)
    z = NNlib.batched_mul(A, grad_x_3d)  # (m, 1, batch)
    spotrs_batched!(L, z, backend)

    # w = Aᵀz
    At = NNlib.batched_transpose(A)
    w = NNlib.batched_mul(At, z)  # (n, 1, batch)

    # grad_A = y(grad_x - w)ᵀ - zxᵀ
    diff = grad_x_3d .- w
    diff_t = NNlib.batched_transpose(diff)
    grad_A = NNlib.batched_mul(y, diff_t)  # (m, n, batch)

    x_t = NNlib.batched_transpose(x)
    temp = NNlib.batched_mul(z, x_t)  # (m, n, batch)
    grad_A .-= temp

    return grad_A, dropdims(z; dims = 2)
end

# ---------------------- QR-based Min-Norm Solver ----------------------

# Union type for QR backends (must match projection.jl)
const QRMinNormBackend = Union{MAGMAQR, cuSOLVERQR}

# Dispatch wrappers for QR operations in min-norm solver
_geqrf_batched!(A, tau, ::MAGMAQR) = magma_sgeqrf_batched!(A, tau)
_geqrf_batched!(A, tau, ::cuSOLVERQR) = cublas_sgeqrf_batched!(A, tau)

_strsm_batched!(side, uplo, trans, diag, alpha, A, B, ::MAGMAQR) =
    magma_strsm_batched!(side, uplo, trans, diag, alpha, A, B)
_strsm_batched!(side, uplo, trans, diag, alpha, A, B, ::cuSOLVERQR) =
    cublas_strsm_batched!(side, uplo, trans, diag, alpha, A, B)

"""
QR-based min-norm forward pass.

Computes x = Aᵀ(AAᵀ)⁻¹b by factorizing Aᵀ = QR, so AAᵀ = RᵀR,
then solving RᵀRy = b via two triangular solves and x = Aᵀy.
Avoids forming the Gram matrix AAᵀ.
"""
function min_norm_forward(A::CuArray{Float32,3}, b::CuArray{Float32,2}, backend::QRMinNormBackend)
    m, n, batch_size = size(A)

    # QR factorize Aᵀ (n×m per batch)
    At = NNlib.batched_transpose(A)        # (n, m, batch) - lazy
    W = permutedims(A, (2, 1, 3))          # (n, m, batch) - contiguous copy of Aᵀ
    tau = CUDA.zeros(Float32, m, batch_size)
    _geqrf_batched!(W, tau, backend)

    # Extract R from upper m×m block (TRSM reads only upper triangle)
    R = W[1:m, :, :]  # (m, m, batch)

    # Solve RᵀRy = b via two triangular solves
    y = reshape(copy(b), m, 1, batch_size)
    _strsm_batched!('L', 'U', 'T', 'N', 1.0f0, R, y, backend)  # Rᵀz = b
    _strsm_batched!('L', 'U', 'N', 'N', 1.0f0, R, y, backend)  # Ry = z

    # x = Aᵀy
    x = NNlib.batched_mul(At, y)  # (n, 1, batch)

    return dropdims(x; dims = 2), MinNormCacheQR(R, y, x)
end

# QR-based backward pass
function min_norm_backward(
    A::CuArray{Float32,3},
    cache::MinNormCacheQR,
    grad_x::CuArray{Float32,2},
    backend::QRMinNormBackend,
)
    m, n, batch_size = size(A)
    (; R, y, x) = cache
    grad_x_3d = reshape(grad_x, n, 1, batch_size)

    # z = (AAᵀ)⁻¹(A·grad_x) via two triangular solves with R
    z = NNlib.batched_mul(A, grad_x_3d)  # (m, 1, batch)
    _strsm_batched!('L', 'U', 'T', 'N', 1.0f0, R, z, backend)  # Rᵀw = A·grad_x
    _strsm_batched!('L', 'U', 'N', 'N', 1.0f0, R, z, backend)  # Rz = w

    # w = Aᵀz
    At = NNlib.batched_transpose(A)
    w = NNlib.batched_mul(At, z)  # (n, 1, batch)

    # grad_A = y(grad_x - w)ᵀ - zxᵀ
    diff = grad_x_3d .- w
    diff_t = NNlib.batched_transpose(diff)
    grad_A = NNlib.batched_mul(y, diff_t)  # (m, n, batch)

    x_t = NNlib.batched_transpose(x)
    temp = NNlib.batched_mul(z, x_t)  # (m, n, batch)
    grad_A .-= temp

    return grad_A, dropdims(z; dims = 2)
end
