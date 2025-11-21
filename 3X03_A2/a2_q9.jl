using Plots

# include("firstname_lastname_a2.jl")
include("Johnson_Ji_a2.jl")

# Fixed 2D anchor positions
P = [-100.0 -50.0 50.0 100.0;
    5.0 -5.0 5.0 -5.0]

# Ground truth location of the receiver
x_gt = [0.0; 50.0]

# Our initial guess for the location of the receiver
x0 = [5.575, -19.297]

newton_final = Vector{Vector{Float64}}()
gradient_final = Vector{Vector{Float64}}()

for i in range(1, 20)

    global d = [norm(P[:, i] - x_gt) + randn() for i in 1:size(P, 2)]

    global x_trace_n = newton_optimizer(x0, P, d, 1e-3, 20)
    global x_trace_g = gradient_descent(x0, P, d, 1e-3, 100, 0.2)

    # save the final location
    push!(newton_final, x_trace_n[end])
    push!(gradient_final, x_trace_g[end])
end

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

# Graph with all of the solutions
p = contourf(X1, X2, log10.(z1), fill=true, color=:turbo,
    lw=0.0, aspect_ratio=:equal, legend=:bottomleft)
scatter!(p, [x[1] for x in newton_final], [x[2] for x in newton_final],
    label="Newton solutions", marker=:o, color=:white)
scatter!(p, [x[1] for x in gradient_final], [x[2] for x in gradient_final],
    label="Gradient solutions", marker=:x, color=:green)
scatter!(p, [x_gt[1]], [x_gt[2]], label="Ground Truth", marker=:x, color=:red, markersize=9)
scatter!(p, P[1, :], P[2, :], label="Beacons", marker=:o, color=:blue)
title!(p, "Solutions found by Newton and Gradient Methods")
savefig(p, "a2_q9_all_solutions.png")

# Graph with trace for final solution
p2 = contourf(X1, X2, log10.(z1), fill=true, color=:turbo,
    lw=0.0, aspect_ratio=:equal, legend=:bottomleft)
plot!(p2, [x[1] for x in x_trace_n], [x[2] for x in x_trace_n],
    label="Newton iterations", marker=:o, color=:white)
plot!(p2, [x[1] for x in x_trace_g], [x[2] for x in x_trace_g],
    label="Gradient iterations", marker=:x, color=:green)
scatter!(p2, [x_gt[1]], [x_gt[2]], label="Ground Truth", marker=:x, color=:red, markersize=9)
scatter!(p2, P[1, :], P[2, :], label="Beacons", marker=:o, color=:blue)
title!("Iterations of Newton and Gradient Methods")
savefig(p2, "a2_q9_one_sol_with_iterations.png")
