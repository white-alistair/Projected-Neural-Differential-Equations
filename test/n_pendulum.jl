@testitem "NPendulum constraints jacobian" begin
    using Random, ForwardDiff, LinearAlgebra, MLDataDevices
    Random.seed!(1)
    
    for device in [MLDataDevices.CPUDevice(), MLDataDevices.CUDADevice()]
        for N in [2, 4, 8]
            T = Float32
            b = 0.1f0  # friction coefficient
            system = ProjectedNDEs.NPendulum{T,N}(b)
            
            # Test that the analytic Jacobian matches the ForwardDiff version
            u = randn(T, 4*N, 1)
            t = 0.0f0
            
            # Reshape to matrix form for the jacobian function
            result = ProjectedNDEs.constraints_jacobian(u, t, system)
            result = result[:, :, 1]  # Extract the single batch result
            
            # Verify output type is Float32
            @test eltype(result) == Float32
            
            # Compute expected Jacobian using ForwardDiff
            expected = ForwardDiff.jacobian(u -> ProjectedNDEs.constraints(u, nothing, system), u)
            
            @test size(result) == (2*N, 4*N)
            @test result ≈ expected atol=1e-6
            
            # Test batched implementation
            batch_size = 4
            U = randn(T, 4*N, batch_size)
            result_batched = ProjectedNDEs.constraints_jacobian(U, t, system)
            
            @test size(result_batched) == (2*N, 4*N, batch_size)
            @test eltype(result_batched) == Float32
            
            # Check each batch element matches the scalar version
            for (i, u_col) in enumerate(eachcol(U))
                u_single = reshape(u_col, 4*N, 1)
                result_single = ProjectedNDEs.constraints_jacobian(u_single, t, system)[:, :, 1]
                @test result_single ≈ result_batched[:, :, i] atol=1e-6
            end
        end
    end
end

@testitem "NPendulum constraints jacobian Chain Rule CPU" begin
    using Random, ChainRulesCore, ChainRulesTestUtils
    Random.seed!(1) 
    T = Float64
    N = 2
    b = 0.1
    system = ProjectedNDEs.NPendulum{T,N}(b)
    
    # Test with batch size 3
    batch_size = 3
    u = randn(T, 4*N, batch_size)
    t = 0.0
    
    # Test the rrule using ChainRulesTestUtils
    test_rrule(ProjectedNDEs.constraints_jacobian, u, t, system ⊢ NoTangent(); rtol=1e-6, atol=1e-8)
end

@testitem "NPendulum constraints jacobian Chain Rule GPU" tags=[:gpu] begin
    using CUDA, Random, ChainRulesTestUtils, ChainRulesCore
    Random.seed!(1) 
    T = Float32
    N = 2
    b = 0.1f0
    system = ProjectedNDEs.NPendulum{T,N}(b)
    
    batch_size = 3
    u_gpu = CUDA.randn(T, 4*N, batch_size)
    t = 0.0f0
    
    # Note: We cannot use ChainRulesTestUtils.test_rrule directly on GPU arrays because
    # FiniteDifferences.jl (used internally by test_rrule) is incompatible with GPU arrays.

    # Test chain rule implementation manually
    # Get the forward pass result and pullback function for GPU
    y_gpu, pullback_gpu = ChainRulesCore.rrule(ProjectedNDEs.constraints_jacobian, u_gpu, t, system)
    
    # Get CPU chain rule for comparison
    u_cpu = Array(u_gpu)
    y_cpu, pullback_cpu = ChainRulesCore.rrule(ProjectedNDEs.constraints_jacobian, u_cpu, t, system)
    @test Array(y_gpu) ≈ y_cpu rtol=1e-6  # Forward passes should match
    
    # Test pullback with same cotangent on both CPU and GPU
    ∂y_cpu = randn(T, size(y_cpu)...)
    ∂y_gpu = CUDA.CuArray(∂y_cpu)  # Use same cotangent values
    
    # Compute gradients on both devices
    _, ∂u_gpu, _, _ = pullback_gpu(∂y_gpu)
    _, ∂u_cpu, _, _ = pullback_cpu(∂y_cpu)
    
    # Compare CPU and GPU gradients - this is the key test!
    @test Array(∂u_gpu) ≈ ∂u_cpu rtol=1e-5
    
    # Verify pullback output properties
    @test ∂u_gpu isa CUDA.CuArray{T,2}
    @test size(∂u_gpu) == size(u_gpu)
    @test eltype(∂u_gpu) == T
end
