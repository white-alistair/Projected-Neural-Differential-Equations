# projection.jl - Project vectors onto constraint nullspace
# Uses the approach: solve GGᵀλ = Gv, then compute v - Gᵀλ

# -------------------------------- Cache Types --------------------------------

# Union type for preconditioner data (diagonal only)
const PreconditionerData = Union{Nothing, CuArray{Float32,2}}

"""
Cache for projection backward pass (Cholesky-based).
Stores Cholesky factor L and solution λ to GGᵀλ = Gv.
Optionally stores preconditioner data for backward pass.
"""
struct ProjectionCache
    L::CuArray{Float32,3}                # Cholesky factor of GGᵀ (or preconditioned)
    λ::CuArray{Float32,3}                # Solution to GGᵀλ = Gv
    preconditioner::PreconditionerData   # Diagonal D⁻¹/² or nothing
end

"""
Cache for projection backward pass (QR-based).
Stores the upper triangular R factor and solution λ.
"""
struct ProjectionCacheQR
    R::CuArray{Float32,3}     # Upper triangular R factor (m, m, batch)
    λ::CuArray{Float32,3}     # Solution λ (m, 1, batch)
end

# ------------------------- Diagonal Preconditioning -------------------------

"""
    compute_symmetric_preconditioner(A::CuArray{Float32,3}; min_diag=1f-8, max_scale=1f3)

Compute D⁻¹/² where D = diag(A) for symmetric preconditioning.
Returns D_inv_sqrt with shape (m, batch).

For numerical stability:
- Diagonal values are clamped to a minimum of `min_diag`
- Preconditioner values are clamped to a maximum of `max_scale`
"""
function compute_symmetric_preconditioner(
    A::CuArray{Float32,3};
    min_diag::Float32=1f-8,
    max_scale::Float32=1f3,
)
    m = size(A, 1)
    batch = size(A, 3)
    # Extract diagonal: A[i,i,b] for each i,b
    # In column-major order, diagonal indices are 1, m+2, 2m+3, ... = 1:(m+1):m*m
    diag_indices = 1:(m+1):m*m
    A_diag = reshape(A, m * m, batch)[diag_indices, :]  # (m, batch)
    # Compute D⁻¹/² with clamping for numerical stability
    D_inv_sqrt = min.(1.0f0 ./ sqrt.(max.(A_diag, min_diag)), max_scale)
    return D_inv_sqrt
end

"""
    apply_symmetric_preconditioner!(A, b, D_inv_sqrt)

Apply symmetric diagonal preconditioning in-place:
  Ã = D⁻¹/² A D⁻¹/²  (preserves symmetry)
  b̃ = D⁻¹/² b

This transforms the system Ax = b into Ãỹ = b̃ where x = D⁻¹/² ỹ.
"""
function apply_symmetric_preconditioner!(
    A::CuArray{Float32,3},
    b::CuArray{Float32,3},
    D_inv_sqrt::CuArray{Float32,2},
)
    m = size(A, 1)
    D_inv_sqrt_row = reshape(D_inv_sqrt, m, 1, :)  # (m, 1, batch)
    D_inv_sqrt_col = reshape(D_inv_sqrt, 1, m, :)  # (1, m, batch)
    
    # Ã[i,j,b] = D_inv_sqrt[i,b] * A[i,j,b] * D_inv_sqrt[j,b]
    A .*= D_inv_sqrt_row
    A .*= D_inv_sqrt_col
    
    # b̃[i,1,b] = D_inv_sqrt[i,b] * b[i,1,b]
    b .*= D_inv_sqrt_row
    
    return nothing
end

"""
    back_transform_solution!(x, D_inv_sqrt)

Back-transform the solution from preconditioned system: x = D⁻¹/² ỹ
"""
function back_transform_solution!(x::CuArray{Float32,3}, D_inv_sqrt::CuArray{Float32,2})
    m = size(x, 1)
    x .*= reshape(D_inv_sqrt, m, 1, :)
    return nothing
end

# ------------------------------ Function Stubs ------------------------------

"""
    project_forward(G, v, backend)

Forward pass for projection onto nullspace of G.
Returns (proj, cache) where proj = v - Gᵀλ and λ solves GGᵀλ = Gv.
"""
function project_forward end

"""
    project_backward(G, cache, grad_proj, proj, backend)

Backward pass for projection.
Returns (grad_G, grad_v).
"""
function project_backward end

# ------------------------------ GPU Public API ------------------------------

"""
    project_to_nullspace(G, v, backend)

Project vector `v` onto the nullspace of constraint Jacobian `G`.

Computes the projection by solving the linear system `GGᵀλ = Gv` for `λ ∈ ℝᵐ`
and then evaluating `Proj(v) = v - Gᵀλ`, which avoids explicitly forming the 
matrix inverse.

# Arguments
- `G`: Constraint Jacobian, size (m, n, batch) where m < n
- `v`: Vector to project, size (n, batch)
- `backend`: Solver backend (MAGMACholesky(), cuSOLVERCholesky(), MAGMAQR(), or cuSOLVERQR())

# Returns  
- Projected vector, size (n, batch)
"""
function project_to_nullspace(
    G::CuArray{Float32,3},
    v::CuArray{Float32,2},
    backend::SolverBackend,
)
    return first(project_forward(G, v, backend))
end

function ChainRulesCore.rrule(
    ::typeof(project_to_nullspace),
    G::CuArray{Float32,3},
    v::CuArray{Float32,2},
    backend::SolverBackend,
)
    proj, cache = project_forward(G, v, backend)
    function pullback(grad_proj)
        grad_G, grad_v = project_backward(G, cache, grad_proj, proj, backend)
        (NoTangent(), grad_G, grad_v, NoTangent())
    end
    return proj, pullback
end

# ------------------------------- CPU Fallback -------------------------------

# CPU project_to_nullspace with backend argument (forwards to no-backend version)
function project_to_nullspace(
    G::AbstractArray{T,3},
    v::AbstractArray{T,2},
    ::SolverBackend,
) where {T}
    return project_to_nullspace(G, v)
end

# CPU project_to_nullspace using the direct GGᵀλ = Gv approach
# Uses reduce/hcat instead of mutation for Zygote compatibility
function project_to_nullspace(
    G::AbstractArray{T,3},
    v::AbstractArray{T,2},
) where {T}
    m, n, batch = size(G)
    slices = map(1:batch) do i
        Gi = G[:, :, i]
        vi = v[:, i]
        # Solve GGᵀλ = Gv
        GGt = Gi * Gi'
        Gv = Gi * vi
        λ = GGt \ Gv
        # proj = v - Gᵀλ
        vi - Gi' * λ
    end
    return reduce(hcat, slices)
end

# ---------------------- GPU Forward Pass (Cholesky) ----------------------

# MAGMA Cholesky forward
function project_forward(G::CuArray{Float32,3}, v::CuArray{Float32,2}, backend::MAGMACholesky)
    m, n, batch = size(G)
    v_3d = reshape(v, n, 1, batch)

    # Compute Gv
    Gv = NNlib.batched_mul(G, v_3d)  # (m, 1, batch)

    # Form A = GGᵀ
    Gt = NNlib.batched_transpose(G)  # (n, m, batch)
    A = NNlib.batched_mul(G, Gt)     # (m, m, batch)

    # Solve GGᵀλ = Gv using combined Cholesky factorization + solve
    λ = copy(Gv)

    # Apply preconditioning if enabled
    preconditioner_data = if backend.preconditioner == :diagonal
        D_inv_sqrt = compute_symmetric_preconditioner(A)
        apply_symmetric_preconditioner!(A, λ, D_inv_sqrt)
        D_inv_sqrt
    else
        nothing
    end

    magma_sposv_batched!(A, λ)  # A now contains Cholesky factor, λ contains solution ỹ

    # Back-transform solution based on preconditioner type
    if preconditioner_data isa CuArray{Float32,2}
        back_transform_solution!(λ, preconditioner_data)
    end

    # Compute proj = v - Gᵀλ
    Gt_lambda = NNlib.batched_mul(Gt, λ)  # (n, 1, batch)
    proj = v .- dropdims(Gt_lambda; dims = 2)

    return proj, ProjectionCache(A, λ, preconditioner_data)
end

# Generic Cholesky forward (cuSOLVER and other backends)
function project_forward(G::CuArray{Float32,3}, v::CuArray{Float32,2}, backend::SolverBackend)
    m, n, batch = size(G)
    v_3d = reshape(v, n, 1, batch)

    # Compute Gv
    Gv = NNlib.batched_mul(G, v_3d)  # (m, 1, batch)

    # Form A = GGᵀ
    Gt = NNlib.batched_transpose(G)  # (n, m, batch)
    A = NNlib.batched_mul(G, Gt)     # (m, m, batch)

    # Solve LLᵀλ = Gv
    λ = copy(Gv)

    # Apply preconditioning if enabled
    preconditioner_type = hasproperty(backend, :preconditioner) ? backend.preconditioner : :none
    preconditioner_data = if preconditioner_type == :diagonal
        D_inv_sqrt = compute_symmetric_preconditioner(A)
        apply_symmetric_preconditioner!(A, λ, D_inv_sqrt)
        D_inv_sqrt
    else
        nothing
    end

    # Cholesky factorization: A where GGᵀ = LLᵀ (in-place)
    spotrf_batched!(A, backend)

    # Solve using Cholesky factor
    spotrs_batched!(A, λ, backend)

    # Back-transform solution based on preconditioner type
    if preconditioner_data isa CuArray{Float32,2}
        back_transform_solution!(λ, preconditioner_data)
    end

    # Compute proj = v - Gᵀλ
    Gt_lambda = NNlib.batched_mul(Gt, λ)  # (n, 1, batch)
    proj = v .- dropdims(Gt_lambda; dims = 2)

    return proj, ProjectionCache(A, λ, preconditioner_data)
end

# ---------------------- GPU Forward Pass (QR) ----------------------

# Dispatch wrappers for QR backends
const QRBackend = Union{MAGMAQR, cuSOLVERQR}

geqrf_batched!(A, tau, ::MAGMAQR) = magma_sgeqrf_batched!(A, tau)
geqrf_batched!(A, tau, ::cuSOLVERQR) = cublas_sgeqrf_batched!(A, tau)

strsm_batched!(side, uplo, trans, diag, alpha, A, B, ::MAGMAQR) =
    magma_strsm_batched!(side, uplo, trans, diag, alpha, A, B)
strsm_batched!(side, uplo, trans, diag, alpha, A, B, ::cuSOLVERQR) =
    cublas_strsm_batched!(side, uplo, trans, diag, alpha, A, B)

"""
QR-based forward pass for nullspace projection.

Factorizes Gᵀ = QR where Q is n×m, R is m×m upper triangular.
Since GGᵀ = RᵀR, solves RᵀRλ = Gv via two triangular solves,
avoiding explicit formation of the Gram matrix GGᵀ.
"""
function project_forward(G::CuArray{Float32,3}, v::CuArray{Float32,2}, backend::QRBackend)
    m, n, batch = size(G)
    v_3d = reshape(v, n, 1, batch)

    # 1. Copy Gᵀ to work array and compute QR factorization
    Gt = NNlib.batched_transpose(G)        # (n, m, batch) - lazy transpose
    W = permutedims(G, (2, 1, 3))          # (n, m, batch) - contiguous copy of Gᵀ
    tau = CUDA.zeros(Float32, m, batch)     # Householder scalars

    geqrf_batched!(W, tau, backend)  # W now contains QR factors

    # 2. Extract R from the upper m×m triangle of W
    #    TRSM with uplo='U' only reads the upper triangle, so we can just slice
    R = W[1:m, :, :]  # (m, m, batch) - contains R in upper triangle

    # 3. Compute Gv
    Gv = NNlib.batched_mul(G, v_3d)  # (m, 1, batch)

    # 4. Solve RᵀRλ = Gv via two triangular solves (in-place on λ)
    λ = copy(Gv)
    # Solve Rᵀz = Gv  (R is upper triangular, so Rᵀ is lower triangular)
    strsm_batched!('L', 'U', 'T', 'N', 1.0f0, R, λ, backend)
    # Solve Rλ = z
    strsm_batched!('L', 'U', 'N', 'N', 1.0f0, R, λ, backend)

    # 5. Compute proj = v - Gᵀλ
    Gt_lambda = NNlib.batched_mul(Gt, λ)  # (n, 1, batch)
    proj = v .- dropdims(Gt_lambda; dims = 2)

    return proj, ProjectionCacheQR(R, λ)
end

# ---------------------- GPU Backward Pass ----------------------

# Cholesky-based backward pass (works for MAGMACholesky, cuSOLVERCholesky)
function project_backward(
    G::CuArray{Float32,3},
    cache::ProjectionCache,
    grad_proj::CuArray{Float32,2},
    proj::CuArray{Float32,2},
    backend::SolverBackend,
)
    m, n, batch = size(G)
    (; L, λ, preconditioner) = cache
    grad_proj_3d = reshape(grad_proj, n, 1, batch)
    proj_3d = reshape(proj, n, 1, batch)

    # Compute μ = (GGᵀ)⁻¹(G · grad_proj) using cached Cholesky factor
    Gt = NNlib.batched_transpose(G)  # (n, m, batch)
    μ = NNlib.batched_mul(G, grad_proj_3d)  # (m, 1, batch)

    # Apply preconditioning to RHS based on preconditioner type
    if preconditioner isa CuArray{Float32,2}
        # Diagonal preconditioning: μ̃_rhs = D⁻¹/² (G · grad_proj)
        μ .*= reshape(preconditioner, m, 1, :)
    end

    # Solve using preconditioned Cholesky factor: L L^T μ̃ = μ̃_rhs
    spotrs_batched!(L, μ, backend)

    # Back-transform based on preconditioner type
    if preconditioner isa CuArray{Float32,2}
        # Diagonal: μ = D⁻¹/² μ̃
        μ .*= reshape(preconditioner, m, 1, :)
    end

    # grad_v = grad_proj - Gᵀμ
    Gt_mu = NNlib.batched_mul(Gt, μ)  # (n, 1, batch)
    grad_v = grad_proj .- dropdims(Gt_mu; dims = 2)

    # grad_G = -μ · projᵀ - λ · grad_vᵀ
    proj_t = NNlib.batched_transpose(proj_3d)  # (1, n, batch)
    grad_v_3d = reshape(grad_v, n, 1, batch)
    grad_v_t = NNlib.batched_transpose(grad_v_3d)  # (1, n, batch)

    grad_G = NNlib.batched_mul(μ, proj_t)  # (m, n, batch)
    grad_G .*= -1
    temp = NNlib.batched_mul(λ, grad_v_t)  # (m, n, batch)
    grad_G .-= temp

    return grad_G, grad_v
end

# QR-based backward pass
"""
QR-based backward pass for nullspace projection.

Uses the cached R factor to solve for μ = (GGᵀ)⁻¹(G · grad_proj)
via two triangular solves with R, then computes gradients.
"""
function project_backward(
    G::CuArray{Float32,3},
    cache::ProjectionCacheQR,
    grad_proj::CuArray{Float32,2},
    proj::CuArray{Float32,2},
    backend::QRBackend,
)
    m, n, batch = size(G)
    (; R, λ) = cache
    grad_proj_3d = reshape(grad_proj, n, 1, batch)
    proj_3d = reshape(proj, n, 1, batch)

    # Compute μ = (GGᵀ)⁻¹(G · grad_proj) = R⁻¹R⁻ᵀ(G · grad_proj)
    Gt = NNlib.batched_transpose(G)  # (n, m, batch)
    μ = NNlib.batched_mul(G, grad_proj_3d)  # (m, 1, batch)

    # Solve RᵀRμ = G·grad_proj via two triangular solves
    # Solve Rᵀw = G·grad_proj
    strsm_batched!('L', 'U', 'T', 'N', 1.0f0, R, μ, backend)
    # Solve Rμ = w
    strsm_batched!('L', 'U', 'N', 'N', 1.0f0, R, μ, backend)

    # grad_v = grad_proj - Gᵀμ
    Gt_mu = NNlib.batched_mul(Gt, μ)  # (n, 1, batch)
    grad_v = grad_proj .- dropdims(Gt_mu; dims = 2)

    # grad_G = -μ · projᵀ - λ · grad_vᵀ
    proj_t = NNlib.batched_transpose(proj_3d)  # (1, n, batch)
    grad_v_3d = reshape(grad_v, n, 1, batch)
    grad_v_t = NNlib.batched_transpose(grad_v_3d)  # (1, n, batch)

    grad_G = NNlib.batched_mul(μ, proj_t)  # (m, n, batch)
    grad_G .*= -1
    temp = NNlib.batched_mul(λ, grad_v_t)  # (m, n, batch)
    grad_G .-= temp

    return grad_G, grad_v
end

