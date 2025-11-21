using Plots

include("Johnson_Ji_a2.jl")

# Fixed 2D anchor positions
P = [0.0 0.0 100.0 100.0;
    0.0 100.0 0.0 100.0]

# Ground truth location of the receiver
x_gt = [20.0; 30.0]

# Compute the noisy distances between the anchors and the receiver
d = [norm(P[:, i] - x_gt) + randn() for i in 1:size(P, 2)]

# Our initial guess for the location of the receiver
x0 = [40.0; 80.0]

# Run your Newton's method-based optimizer
x_trace = newton_optimizer(x0, P, d, 1e-3, 20)

# Plot the results on a contour map
x1 = [x[1] for x in x_trace]
x2 = [x[2] for x in x_trace]

limit = 100.0
N_plot = 100
X1 = range(-limit, stop=limit, length=N_plot)
X2 = range(-limit, stop=limit, length=N_plot)

# Define the cost function
function f(x, P, d)
    cost = 0.0
    for i in 1:size(P, 2)
        cost += (norm(x - P[:, i]) - d[i])^2
    end
    return cost
end

# Create a modified cost function for ease of 2D plotting
f_plot(x1, x2) = f([x1; x2], P, d)
z1 = @. f_plot(X1', X2)
contourf(X1, X2, log10.(z1), fill=true, color=:turbo,
    lw=0.0, aspect_ratio=:equal, legend=:bottomleft)
plot!(x1, x2, label="Newton Iterations", marker=:o, color=:white)
scatter!([x_gt[1]], [x_gt[2]], label="Ground Truth", marker=:x, color=:red, markersize=9)
scatter!(P[1, :], P[2, :], label="Beacons")
title!("log10 of Cost")
savefig("newton_optimizer_plot.png")

# ---- P7 test ----
x_hat = x_trace[end]
n = length(x_hat)

gradient_f(xc) = begin
    g = zeros(n)
    m = size(P, 2)
    @inbounds for i in 1:m
        dx = xc .- P[:, i]
        nd = norm(dx)
        if nd > 1e-12
            g .+= 2 * (nd - d[i]) * (dx / nd)
        end
    end
    g
end

hessian_f(xc) = begin
    H = zeros(n, n)
    m = size(P, 2)
    I_n = Matrix{Float64}(I, n, n)
    @inbounds for i in 1:m
        dx = xc .- P[:, i]
        nd = norm(dx)
        if nd > 1e-12
            term1 = (dx * dx') / nd^2
            term2 = (nd - d[i]) * ((I_n / nd) - (dx * dx') / nd^3)
            H .+= 2 * (term1 + term2)
        end
    end
    H
end

g_final = gradient_f(x_hat)
println("\n--- Problem 7 Test (same data as plot) ---")
println("Estimated x̂       = ", x_hat)
println("Ground truth x_gt = ", x_gt)
println("Final gradient norm ||∇f(x̂)|| = ", norm(g_final))
println("Final gradient = ", g_final)

H_final = hessian_f(x_hat)
eigvals_H = eigvals(H_final)
println("Eigenvalues of Hessian at x̂: ", eigvals_H)
println("λ_min = ", minimum(eigvals_H), ", λ_max = ", maximum(eigvals_H))
println("--------------------------------------------------")
