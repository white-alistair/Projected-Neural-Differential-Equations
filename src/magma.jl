# magma.jl - Batched minimum-norm solver using MAGMA

const MAGMA_LIB = get(ENV, "MAGMA_LIB", "libmagma.so")
const MAGMA_QUEUE = Ref{Ptr{Cvoid}}(C_NULL)

const MagmaLower = 122
const MagmaUpper = 121
const MagmaLeft = 141
const MagmaTrans = 112
const MagmaNoTrans = 111
const MagmaNonUnit = 131

function magma_init()
    isfile(MAGMA_LIB) || error("MAGMA not found at $MAGMA_LIB. Set ENV[\"MAGMA_LIB\"].")
    ccall((:magma_init, MAGMA_LIB), Cint, ()) == 0 || error("magma_init failed")
    ccall(
        (:magma_queue_create_internal, MAGMA_LIB),
        Cvoid,
        (Cint, Ref{Ptr{Cvoid}}, Cstring, Cstring, Cint),
        0,
        MAGMA_QUEUE,
        "magma.jl",
        "magma_init",
        0,
    )
end

function magma_queue_sync()
    ccall(
        (:magma_queue_sync_internal, MAGMA_LIB),
        Cvoid,
        (Ptr{Cvoid}, Cstring, Cstring, Cint),
        MAGMA_QUEUE[],
        "magma.jl",
        "sync",
        0,
    )
    # Synchronize Julia's CUDA stream to ensure MAGMA completes before Julia CUDA ops continue
    # This is needed because MAGMA uses a separate queue from Julia's CUDA stream
    CUDA.synchronize()
end


# ---------------------- MAGMA Cholesky Bindings ----------------------

# Batched Cholesky factorization
function magma_spotrf_batched!(A::CuArray{Float32,3})
    m, n, batch_size = size(A)
    @assert m == n "Matrix must be square"

    lda = max(1, stride(A, 2))
    Aptrs = unsafe_strided_batch(A)
    info = CuArray{Cint}(undef, batch_size)  # We don't actually use this

    GC.@preserve A Aptrs info begin
        ccall(
            (:magma_spotrf_batched, MAGMA_LIB),
            Cint,
            (Cint, Cint, CuPtr{Ptr{Cfloat}}, Cint, CuPtr{Cint}, Cint, Ptr{Cvoid}),
            MagmaLower,
            n,
            Aptrs,
            lda,
            info,
            batch_size,
            MAGMA_QUEUE[],
        )
        magma_queue_sync()
    end

    CUDA.unsafe_free!(Aptrs)
    CUDA.unsafe_free!(info)
    return A
end

# Batched linear solve using Cholesky factorization
function magma_spotrs_batched!(L::CuArray{Float32,3}, B::CuArray{Float32,3})
    m, n, batch_size = size(L)

    nrhs = size(B, 2)

    Lptrs = unsafe_strided_batch(L)
    Bptrs = unsafe_strided_batch(B)
    ldl = max(1, stride(L, 2))
    ldb = max(1, stride(B, 2))

    GC.@preserve L B Lptrs Bptrs begin
        ccall(
            (:magma_spotrs_batched, MAGMA_LIB),
            Cint,
            (
                Cint,
                Cint,
                Cint,
                CuPtr{Ptr{Cfloat}},
                Cint,
                CuPtr{Ptr{Cfloat}},
                Cint,
                Cint,
                Ptr{Cvoid},
            ),
            MagmaLower,
            n,
            nrhs,
            Lptrs,
            ldl,
            Bptrs,
            ldb,
            batch_size,
            MAGMA_QUEUE[],
        )
        magma_queue_sync()
    end

    CUDA.unsafe_free!(Lptrs)
    CUDA.unsafe_free!(Bptrs)
    return B
end

# ---------------------- MAGMA Cholesky Combined ----------------------

# Batched Cholesky factorization + solve (combined)
function magma_sposv_batched!(A::CuArray{Float32,3}, B::CuArray{Float32,3})
    m, n, batch_size = size(A)
    @assert m == n "Matrix must be square"

    nrhs = size(B, 2)

    Aptrs = unsafe_strided_batch(A)
    Bptrs = unsafe_strided_batch(B)
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    info = CuArray{Cint}(undef, batch_size)

    GC.@preserve A B Aptrs Bptrs info begin
        ccall(
            (:magma_sposv_batched, MAGMA_LIB),
            Cint,
            (
                Cint,
                Cint,
                Cint,
                CuPtr{Ptr{Cfloat}},
                Cint,
                CuPtr{Ptr{Cfloat}},
                Cint,
                CuPtr{Cint},
                Cint,
                Ptr{Cvoid},
            ),
            MagmaLower,
            n,
            nrhs,
            Aptrs,
            lda,
            Bptrs,
            ldb,
            info,
            batch_size,
            MAGMA_QUEUE[],
        )
        magma_queue_sync()
    end

    CUDA.unsafe_free!(Aptrs)
    CUDA.unsafe_free!(Bptrs)
    CUDA.unsafe_free!(info)
    return A, B
end

# ---------------------- MAGMA QR Bindings ----------------------

# Batched QR factorization
# A is (rows, cols, batch), tau is (min(rows,cols), batch)
# A is overwritten: R in upper triangle, Householder reflectors below diagonal
function magma_sgeqrf_batched!(A::CuArray{Float32,3}, tau::CuArray{Float32,2})
    rows, cols, batch_size = size(A)

    lda = max(1, stride(A, 2))
    Aptrs = unsafe_strided_batch(A)
    tau_ptrs = unsafe_strided_batch(tau)
    info = CuArray{Cint}(undef, batch_size)

    GC.@preserve A tau Aptrs tau_ptrs info begin
        ccall(
            (:magma_sgeqrf_batched, MAGMA_LIB),
            Cint,
            (
                Cint,
                Cint,
                CuPtr{Ptr{Cfloat}},
                Cint,
                CuPtr{Ptr{Cfloat}},
                CuPtr{Cint},
                Cint,
                Ptr{Cvoid},
            ),
            rows,
            cols,
            Aptrs,
            lda,
            tau_ptrs,
            info,
            batch_size,
            MAGMA_QUEUE[],
        )
        magma_queue_sync()
    end

    CUDA.unsafe_free!(Aptrs)
    CUDA.unsafe_free!(tau_ptrs)
    CUDA.unsafe_free!(info)
    return A, tau
end

# Batched triangular solve: op(A) * X = alpha * B  (side='L')
# A is (m, m, batch) triangular, B is (m, nrhs, batch), solved in-place on B
# side, uplo, trans, diag are MAGMA constants (Cint)
function magma_strsm_batched!(
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

    # Convert Char to MAGMA constants
    magma_side  = side  == 'L' ? MagmaLeft : Cint(142)  # MagmaRight
    magma_uplo  = uplo  == 'U' ? MagmaUpper : MagmaLower
    magma_trans = trans == 'T' ? MagmaTrans : MagmaNoTrans
    magma_diag  = diag  == 'N' ? MagmaNonUnit : Cint(132)  # MagmaUnit

    GC.@preserve A B Aptrs Bptrs begin
        ccall(
            (:magmablas_strsm_batched, MAGMA_LIB),
            Cvoid,
            (
                Cint,
                Cint,
                Cint,
                Cint,
                Cint,
                Cint,
                Cfloat,
                CuPtr{Ptr{Cfloat}},
                Cint,
                CuPtr{Ptr{Cfloat}},
                Cint,
                Cint,
                Ptr{Cvoid},
            ),
            magma_side,
            magma_uplo,
            magma_trans,
            magma_diag,
            m_B,
            n_B,
            alpha,
            Aptrs,
            lda,
            Bptrs,
            ldb,
            batch_size,
            MAGMA_QUEUE[],
        )
        magma_queue_sync()
    end

    CUDA.unsafe_free!(Aptrs)
    CUDA.unsafe_free!(Bptrs)
    return B
end
