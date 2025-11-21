using LinearAlgebra
#Problem 1
function power_method_symmetric(A, tol)
    n = size(A, 1)
    v = rand(n)
    v /= norm(v)
    λ = dot(v, A * v)
    w = A * v
    r = norm(w - λ * v)
    it = 0
    while r > tol && it < 10_000 # the power method just multiply the matrix by many times to extract the domiance eigenvector's direction
        v = w / norm(w) # normalize the vector
        λ = dot(v, A * v) # Rayleigh quotient to estimate the eigenvalue
        w = A * v
        r = norm(w - λ * v) # residual to check convergence
        it += 1
    end
    return λ, v
end
#Problem 3
function extremal_eigenpairs(A, k, tol)
    n = size(A, 1)
    Ak = Matrix{Float64}(A)
    λ = Vector{Float64}(undef, k)
    V = Matrix{Float64}(undef, n, k)
    max_iters = 10_000
    for j in 1:k
        v = randn(n)
        v ./= norm(v)
        λj = 0.0
        for _ in 1:max_iters # extract the largest eigenvalue and its eigenvector
            w = Ak * v # multiply matrix by vector
            nw = norm(w) # compute the norm
            if nw == 0.0
                λj = 0.0
                break
            end
            v = w / nw # normalize the vector
            λj = dot(v, Ak * v) # Rayleigh quotient to estimate the eigenvalue
            r = Ak * v .- λj * v # residual to check convergence
            if norm(r) <= tol
                break
            end
        end
        λ[j] = λj
        V[:, j] = v
        Ak .-= λj * (v * v')
    end
    return λ, V
end

#Problem 5
function newton(x0, P, d, tol, max_iters)
    n = length(x0)
    x = Vector{Float64}(x0)
    x_trace = Vector{Vector{Float64}}()
    push!(x_trace, copy(x))
    ϵ = 1e-12 # small constant to avoid division by zero to get the error
    for _ in 1:max_iters
        r = similar(d, n)
        J = Matrix{Float64}(undef, n, n)
        @inbounds for i in 1:n
            dx = x .- @view P[:, i] # difference vector between current estimate and point P[:, i]
            nr = sqrt(sum(abs2, dx)) # the second norm
            s = max(nr, ϵ) # avoid division by zero
            r[i] = nr - d[i] # residual for the i-th distance constraint
            J[i, :] = (dx ./ s)' # Jacobian matrix row for the i constraint
        end
        if norm(r) <= tol # stop
            break
        end
        Δ = J \ (-r) # solve for the update step
        x .+= Δ # update the estimate
        push!(x_trace, copy(x))
        if norm(Δ) <= tol
            break
        end
    end
    return x_trace
end
#Problem 7
function newton_optimizer(x0, P, d, tol, max_iters)
    n = length(x0)
    x = copy(Vector{Float64}(x0))
    x_trace = Vector{Vector{Float64}}()
    push!(x_trace, copy(x))
    gradient_f(xc) = begin
        g = zeros(n)
        m = size(P, 2)
        @inbounds for i in 1:m
            dx = xc .- P[:, i]
            nd = norm(dx)
            if nd > 1e-12
                g .+= 2 * ((nd - d[i]) * (dx / nd)) # compute the gradient
            end
        end
        g
    end
    hessian_f(xc) = begin # compute the Hessian matrix
        H = zeros(n, n)
        m = size(P, 2)
        I_n = Matrix{Float64}(I, n, n)
        @inbounds for i in 1:m
            dx = xc .- P[:, i]
            nd = norm(dx)
            if nd > 1e-12
                term1 = (dx * dx') / (nd^2)
                term2 = (nd - d[i]) * ((I_n / nd) - (dx * dx') / (nd^3))
                H .+= 2 * (term1 + term2)
            end
        end
        H
    end
    solve_lin!(A, b) = begin
        m = size(A, 1)
        A = copy(A)
        b = copy(b)
        for k in 1:m-1
            p = k
            mv = abs(A[k, k])
            for r in k+1:m
                v = abs(A[r, k])
                if v > mv
                    mv = v
                    p = r
                end
            end
            if p != k
                A[k, :], A[p, :] = A[p, :], A[k, :]
                b[k], b[p] = b[p], b[k]
            end
            akk = A[k, k]
            if abs(akk) < 1e-14
                continue
            end
            for i in k+1:m
                f = A[i, k] / akk
                @inbounds A[i, k:m] .-= f .* A[k, k:m]
                b[i] -= f * b[k]
            end
        end
        xsol = Vector{Float64}(undef, m)
        for i in m:-1:1
            s = b[i]
            @inbounds for j in i+1:m
                s -= A[i, j] * xsol[j]
            end
            den = A[i, i]
            xsol[i] = abs(den) < 1e-14 ? 0.0 : s / den
        end
        xsol
    end
    for _ in 1:max_iters
        g = gradient_f(x)
        if norm(g) <= tol
            break
        end
        H = hessian_f(x)
        s = solve_lin!(H, -g)
        x .= x .+ s
        push!(x_trace, copy(x))
        if norm(s) <= tol
            break
        end
    end
    return x_trace
end
#Problem 8
function gradient_descent(x0, P, d, tol, max_iters, gamma)
    n = length(x0)
    x = copy(Vector{Float64}(x0))
    x_trace = Vector{Vector{Float64}}()
    push!(x_trace, copy(x))
    gradient_f(xc) = begin # compute the gradient
        g = zeros(n)
        m = size(P, 2)
        @inbounds for i in 1:m
            dx = xc .- P[:, i]
            nd = norm(dx) # compute the norm of the difference vector
            if nd > 1e-12
                g .+= 2 * ((nd - d[i]) * (dx / nd))
            end
        end
        g
    end
    for _ in 1:max_iters
        g = gradient_f(x)
        if norm(g) <= tol
            break
        end
        x .-= gamma .* g
        push!(x_trace, copy(x))
    end
    return x_trace
end
