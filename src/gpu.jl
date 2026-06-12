Zygote.@adjoint CUDA.zeros(x...) = CUDA.zeros(x...), _ -> map(_ -> nothing, x)

Base.zeros(::MLDataDevices.CUDADevice, dims...) = CUDA.zeros(dims)
Base.zeros(::MLDataDevices.CPUDevice, dims...) = zeros(dims)

Base.zeros(::MLDataDevices.CUDADevice, T, dims...) = CUDA.zeros(T, dims)
Base.zeros(::MLDataDevices.CPUDevice, T, dims...) = zeros(T, dims)

cpu(x) = MLDataDevices.CPUDevice()(x)

# ---------------------- Pointer Array Helper ----------------------

"""
Create a device array of pointers to each slice of a 3D array.

This is functionally identical to `CUDA.CUBLAS.unsafe_strided_batch` function,
but for some reason it's much faster to define it here. I have no idea why.

We define it here because it's used in both `magma.jl` and `cusolver.jl`.
"""
@inline function unsafe_strided_batch(strided::CUDA.DenseCuArray{T}) where {T}
    batch_size = last(size(strided))
    batch_stride = prod(size(strided)[1:end-1])
    ptrs = CuArray{CuPtr{T}}(undef, batch_size)

    # Compute pointers on the GPU
    function compute_pointers()
        i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
        grid_stride = gridDim().x * blockDim().x
        while i <= length(ptrs)
            @inbounds ptrs[i] = reinterpret(
                CuPtr{T},
                pointer(strided, (i - Int32(1)) * batch_stride + Int32(1)),
            )
            i += grid_stride
        end
        return
    end

    kernel = @cuda launch = false compute_pointers()
    config = launch_configuration(kernel.fun)
    threads = min(config.threads, batch_size)
    blocks = min(config.blocks, cld(batch_size, threads))
    @cuda threads = threads blocks = blocks compute_pointers()

    # Synchronize to ensure pointers are computed before MAGMA (which uses a different queue)
    CUDA.synchronize()
    return ptrs
end
