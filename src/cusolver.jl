# cusolver.jl - Batched minimum-norm solver using cuSOLVER

# Batched Cholesky factorization
function cusolver_spotrf_batched!(A::CuArray{Float32,3})
    m, n, batch_size = size(A)
    @assert m == n "Matrix must be square"

    lda = max(1, stride(A, 2))
    Aptrs = unsafe_strided_batch(A)
    info = CuArray{Cint}(undef, batch_size)  # We don't actually use this

    CUDA.CUSOLVER.cusolverDnSpotrfBatched(
        CUDA.CUSOLVER.dense_handle(),
        'L',
        n,
        Aptrs,
        lda,
        info,
        batch_size,
    )

    CUDA.unsafe_free!(Aptrs)
    CUDA.unsafe_free!(info)
    return A
end

# Batched linear solve using Cholesky factorization
function cusolver_spotrs_batched!(L::CuArray{Float32,3}, B::CuArray{Float32,3})
    m, n, batch_size = size(L)

    Lptrs = unsafe_strided_batch(L)
    Bptrs = unsafe_strided_batch(B)
    ldl = max(1, stride(L, 2))
    ldb = max(1, stride(B, 2))

    info = CuArray{Cint}(undef, 1)  # We don't actually use this

    CUDA.CUSOLVER.cusolverDnSpotrsBatched(
        CUDA.CUSOLVER.dense_handle(),
        'L',
        n,
        1,
        Lptrs,
        ldl,
        Bptrs,
        ldb,
        info,
        batch_size,
    )

    CUDA.unsafe_free!(Lptrs)
    CUDA.unsafe_free!(Bptrs)
    CUDA.unsafe_free!(info)
    return B
end

# ---------------------- CUBLAS QR Bindings ----------------------

# Batched QR factorization using CUBLAS
# A is (rows, cols, batch), tau is (min(rows,cols), batch)
# A is overwritten: R in upper triangle, Householder reflectors below diagonal
function cublas_sgeqrf_batched!(A::CuArray{Float32,3}, tau::CuArray{Float32,2})
    rows, cols, batch_size = size(A)

    lda = max(1, stride(A, 2))
    Aptrs = unsafe_strided_batch(A)
    tau_ptrs = unsafe_strided_batch(tau)
    info = Ref{Cint}()

    CUDA.CUBLAS.cublasSgeqrfBatched(
        CUDA.CUBLAS.handle(),
        rows,
        cols,
        Aptrs,
        lda,
        tau_ptrs,
        info,
        batch_size,
    )

    CUDA.unsafe_free!(Aptrs)
    CUDA.unsafe_free!(tau_ptrs)
    return A, tau
end

# ---------------------- CUBLAS TRSM Binding ----------------------

# Batched triangular solve: op(A) * X = alpha * B  (side='L')
# A is (m, m, batch) triangular, B is (m, nrhs, batch), solved in-place on B
function cublas_strsm_batched!(
    side::Char, uplo::Char, trans::Char, diag::Char,
    alpha::Float32,
    A::CuArray{Float32,3},
    B::CuArray{Float32,3},
)
    m_B = size(B, 1)
    n_B = size(B, 2)
    batch_size = size(B, 3)

    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    Aptrs = unsafe_strided_batch(A)
    Bptrs = unsafe_strided_batch(B)

    CUDA.CUBLAS.cublasStrsmBatched(
        CUDA.CUBLAS.handle(),
        side,
        uplo,
        trans,
        diag,
        m_B,
        n_B,
        alpha,
        Aptrs,
        lda,
        Bptrs,
        ldb,
        batch_size,
    )

    CUDA.unsafe_free!(Aptrs)
    CUDA.unsafe_free!(Bptrs)
    return B
end
