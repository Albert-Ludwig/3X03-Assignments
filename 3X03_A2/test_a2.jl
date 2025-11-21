using Random
using LinearAlgebra
Random.seed!(99991099910101010)

# Replace with your solution:
# include("firstname_lastname_a2.jl")
include("Johnson_Ji_a2.jl")

"""
Use this script to test your solutions (you will be graded by a similar script).
You may find it useful to modify the parameters or add your own tests.

DO NOT SUBMIT THIS FILE WITH YOUR SOLUTION!
"""

n_trials = 10
n = 100

## Test power_method_symmetric accuracy
power_method_score = 0
tol = 1e-9
n_trials = 50
for _ in 1:n_trials
    A = 2.0 * rand(n, n) .- 1.0
    A = A + A'
    λ, v = power_method_symmetric(A, tol)
    eigpairs = eigen(A)
    _, max_ind = findmax(abs.(eigpairs.values))
    val_correct = abs(eigpairs.values[max_ind] - λ) <= tol
    vec_correct = norm(eigpairs.vectors[:, max_ind] - v) <= tol * sqrt(n) ||
                  norm(eigpairs.vectors[:, max_ind] + v) <= tol * sqrt(n)
    global power_method_score += val_correct && vec_correct
end
power_method_score = Int(round(power_method_score / 10.0))
println("Problem 1 score: $(power_method_score)/5 marks")

## Test extremal_eigenpairs accuracy
extremal_eigs_score = 0
for _ in 1:n_trials
    A = 2.0 * rand(n, n) .- 1.0
    A = A + A'
    k = rand(2:5)
    λ, _ = extremal_eigenpairs(A, k, tol)
    eigpairs = eigen(A)
    max_inds = sortperm(abs.(eigpairs.values), rev=true)
    λ_true = eigpairs.values[max_inds[1:k]]
    vals_correct = all(abs.(λ_true - λ) .<= tol)
    global extremal_eigs_score += vals_correct
end
extremal_eigs_score = Int(round(extremal_eigs_score / 10.0))
println("Problem 3 score: $(extremal_eigs_score)/5 marks")


# Test out the Newton solver 
score_newton = 0
newton_tol = 1e-6
for i in 1:n_trials
    x_gt = rand(3) * 10 .- 5.0
    x0 = x_gt + 0.5 * rand(3)
    P = rand(3, 3) * 10.0 .- 5.0
    d = [norm(x_gt - P[:, i]) for i in 1:size(P, 2)]
    x_trace = newton(x0, P, d, newton_tol, 100)
    F = [norm(x_trace[end] - P[:, i]) - d[i] for i in 1:3]
    global score_newton += (norm(F) <= newton_tol)
end
score_newton = Int(round(score_newton / 5.0))
println("Problem 5 score: $(score_newton)/10")

# Test gradient descent
score_newton = 0
newton_tol = 1e-6
for _ in 1:n_trials
    x_gt = rand(3)
    x0 = rand(3)
    P = [0.0 1.0 0.0 1.0;
        0.0 1.0 1.0 0.0;
        0.0 0.0 1.0 1.0]
    d = [norm(x_gt - P[:, i]) for i in 1:size(P, 2)]
    x_trace = gradient_descent(x0, P, d, newton_tol, 100, 0.3)
    global score_newton += (norm(x_trace[end] - x_gt) <= newton_tol)
end
score_newton = Int(round(score_newton * 5 / n_trials))
println("Problem 8.1 score: $(score_newton)/5")