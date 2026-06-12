@testitem "Solver Backends" tags = [:gpu] begin
    using CUDA, Random, LinearAlgebra
    Random.seed!(42)

    m, n, batch = 4, 8, 16

    # Create random constraint Jacobians and vectors on CPU
    G_cpu = randn(Float32, m, n, batch)
    v_cpu = randn(Float32, n, batch)

    # Ground truth: direct computation with Julia's native linalg
    result_cpu = similar(v_cpu)
    for i = 1:batch
        Gi, vi = G_cpu[:, :, i], v_cpu[:, i]
        result_cpu[:, i] = vi - Gi \ (Gi * vi)
    end

    # # Verify result is in nullspace: G * result ≈ 0
    for i = 1:batch
        @test norm(G_cpu[:, :, i] * result_cpu[:, i]) < 1e-5
    end

    # Test CPU backend
    result_cpu_backend = ProjectedNDEs.project_to_nullspace(G_cpu, v_cpu)
    @test result_cpu_backend ≈ result_cpu rtol = 1e-6

    # GPU arrays
    G_gpu = CuArray(G_cpu)
    v_gpu = CuArray(v_cpu)

    # Test cuSOLVERCholesky backend
    result_cusolver =
        ProjectedNDEs.project_to_nullspace(G_gpu, v_gpu, ProjectedNDEs.cuSOLVERCholesky())
    @test Array(result_cusolver) ≈ result_cpu rtol = 1e-6

    # Test MAGMACholesky backend
    result_magma =
        ProjectedNDEs.project_to_nullspace(G_gpu, v_gpu, ProjectedNDEs.MAGMACholesky())
    @test Array(result_magma) ≈ result_cpu rtol = 1e-6

    # Test MAGMAQR backend
    result_magma_qr = ProjectedNDEs.project_to_nullspace(G_gpu, v_gpu, ProjectedNDEs.MAGMAQR())
    @test Array(result_magma_qr) ≈ result_cpu rtol = 1e-6

    # Test cuSOLVERQR backend
    result_cusolver_qr =
        ProjectedNDEs.project_to_nullspace(G_gpu, v_gpu, ProjectedNDEs.cuSOLVERQR())
    @test Array(result_cusolver_qr) ≈ result_cpu rtol = 1e-6
end

@testitem "Solver Gradients" tags = [:gpu] begin
    using CUDA, Random, LinearAlgebra, ChainRulesCore, Zygote

    Random.seed!(42)
    m, n, batch = 4, 8, 16

    # Create random matrix A (m, n, batch) and right-hand side b (m, batch)
    A_cpu = randn(Float32, m, n, batch)
    b_cpu = randn(Float32, m, batch)

    A_gpu = CuArray(A_cpu)
    b_gpu = CuArray(b_cpu)

    # Random upstream gradient for output x which has shape (n, batch)
    grad_out_cpu = randn(Float32, n, batch)
    grad_out_gpu = CuArray(grad_out_cpu)

    # Get reference gradients from CPU using Zygote
    # The CPU fallback uses Julia's native linear algebra which Zygote can differentiate
    result_ref, back_ref =
        Zygote.pullback((A, b) -> ProjectedNDEs.min_norm_solve(A, b), A_cpu, b_cpu)
    grad_A_ref, grad_b_ref = back_ref(grad_out_cpu)

    # Helper to test gradient for a GPU backend against CPU reference
    function test_gradient(backend)
        result_gpu, pullback_gpu =
            ChainRulesCore.rrule(ProjectedNDEs.min_norm_solve, A_gpu, b_gpu, backend)
        _, grad_A_gpu, grad_b_gpu, _ = pullback_gpu(grad_out_gpu)

        # Forward pass should match reference
        @test Array(result_gpu) ≈ result_ref rtol = 1e-6

        # Gradients should match reference
        @test Array(grad_A_gpu) ≈ grad_A_ref rtol = 1e-6
        @test Array(grad_b_gpu) ≈ grad_b_ref rtol = 1e-6
    end

    # Test GPU backends against CPU reference
    test_gradient(ProjectedNDEs.cuSOLVERCholesky())
    test_gradient(ProjectedNDEs.MAGMACholesky())
    test_gradient(ProjectedNDEs.MAGMAQR())
    test_gradient(ProjectedNDEs.cuSOLVERQR())
end

@testitem "Projection Gradients" tags = [:gpu] begin
    using CUDA, Random, LinearAlgebra, ChainRulesCore, Zygote

    Random.seed!(42)
    m, n, batch = 4, 8, 16

    # Create random constraint Jacobians and vectors
    G_cpu = randn(Float32, m, n, batch)
    v_cpu = randn(Float32, n, batch)

    G_gpu = CuArray(G_cpu)
    v_gpu = CuArray(v_cpu)

    # Random upstream gradient for output proj which has shape (n, batch)
    grad_out_cpu = randn(Float32, n, batch)
    grad_out_gpu = CuArray(grad_out_cpu)

    # Get reference gradients from CPU using Zygote
    # The CPU fallback uses Julia's native linear algebra which Zygote can differentiate
    result_ref, back_ref = Zygote.pullback(
        (G, v) -> ProjectedNDEs.project_to_nullspace(G, v), G_cpu, v_cpu)
    grad_G_ref, grad_v_ref = back_ref(grad_out_cpu)

    # Helper to test gradient for a GPU backend against CPU reference
    function test_projection_gradient(backend)
        result_gpu, pullback_gpu = ChainRulesCore.rrule(
            ProjectedNDEs.project_to_nullspace, G_gpu, v_gpu, backend)
        _, grad_G_gpu, grad_v_gpu, _ = pullback_gpu(grad_out_gpu)

        # Forward pass should match reference
        @test Array(result_gpu) ≈ Array(result_ref) rtol = 1e-6

        # Gradients should match reference
        @test Array(grad_G_gpu) ≈ grad_G_ref rtol = 1e-5
        @test Array(grad_v_gpu) ≈ grad_v_ref rtol = 1e-5
    end

    # Test GPU backends against CPU reference
    test_projection_gradient(ProjectedNDEs.cuSOLVERCholesky())
    test_projection_gradient(ProjectedNDEs.MAGMACholesky())
    test_projection_gradient(ProjectedNDEs.MAGMAQR())
    test_projection_gradient(ProjectedNDEs.cuSOLVERQR())
end
