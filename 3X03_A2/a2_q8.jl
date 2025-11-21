using Plots

# include("firstname_lastname_a2.jl")
include("Johnson_Ji_a2.jl")

# Fixed 2D anchor positions
P = [-75.0 -75.0 -75.0 50.0 50.0 50.0;
    25.0 0.0 50.0 25.0 0.0 50.0]

# Ground truth location of the receiver
x_gt = [0.0; 0.0]

# Compute the noisy distances between the anchors and the receiver
d = [norm(P[:, i] - x_gt) + randn() for i in 1:size(P, 2)]

# Our initial guess for the location of the receiver
x0 = [25.0; 70.0]

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

# Values of gamma to test
gammas = [0.05, 0.15, 0.25]

for g in gammas

    # Run your Newton's method-based optimizer
    x_trace = gradient_descent(x0, P, d, 1e-3, 200, g)
    println("Gradient Descent iterations, γ=$(g): $(x_trace.size[1])")

    # Plot the results on a contour map
    local p = contourf(X1, X2, log10.(z1), fill=true, color=:turbo,
        lw=0.0, aspect_ratio=:equal, legend=:bottomleft)
    plot!(p, [x[1] for x in x_trace], [x[2] for x in x_trace],
        label="Gradient Descent Iterations", marker=:o, color=:green)
    scatter!(p, [x_gt[1]], [x_gt[2]], label="Ground Truth", marker=:x, color=:red, markersize=9)
    scatter!(p, P[1, :], P[2, :], label="Beacons", marker=:o, color=:blue)
    title!(p, "Gradient Descent, γ = $(g)")

    # Save plot to a file
    savefig(p, "a2_q8_$(replace(string(g), "." => "_")).png")
end