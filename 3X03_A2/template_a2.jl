using LinearAlgebra

"""
Computes the maximum magnitude eigenpair for the real symmetric matrix A 
with the power method. Ensures that the error tolerance is satisfied by 
using the Bauer-Fike theorem.

Inputs:
    A: real symmetric matrix
    tol: error tolerance; i.e., the maximum eigenvalue estimate 
         must be within tol of the true maximum eigenvalue

Outputs:
    λ: the estimate of the maximum magnitude eigenvalue
    v: the estimate of the eigenvector corresponding to λ
"""
function power_method_symmetric(A, tol)
    return λ, v
end

"""
Compute the eigenpairs of the k extremal eigenvalues (i.e., the k eigenvalues of real 
symmetric matrix A with the greatest absolute value).

Inputs:
    A: nxn real symmetric matrix
    k: number of extremal (i.e., maximum magnitude) eigenvalues to compute
    tol: error tolerance; i.e., each eigenvalue estimate 
         must be within tol of a true eigenvalue

Outputs:
    λ: vector of k real elements containing the estimates of the extremal eigenvalues;
       λ[i] contains the ith largest eigenvalue by absolute value
    V: nxk matrix where V[:, i] is the eigenvector for the ith largest eigenvalue 
       by absolute value

"""
function extremal_eigenpairs(A, k, tol)
    return λ, V
end

"""
Use Newton's method to solve the nonlinear system of equations described in Problems 4-5.
This should work for Euclidean distance measurements in any dimension n.

Inputs:
    x0: initial guess for the position of the receiver in R^n
    P: nxn matrix with known locations of transmitting beacons as columns
    d: vector in R^n where d[i] contains the distance from beacon P[:, i] to x
    tol: Euclidean error tolerance (stop when norm(F(x)) <= tol)
    max_iters: maximum iterations of Newton's method to try

Returns:
    x_trace: Vector{Vector{Float64}} containing each Newton iterate x_k in R^n. 

"""
function newton(x0, P, d, tol, max_iters)
    return x_trace
end

"""
Use Newton's method to solve the nonlinear optimization problem described in Problems 6-7.
This should work for Euclidean distance measurements in any dimension n, and any number 
    of noisy measurements m.

Inputs:
    x0: initial guess for the position of the receiver in R^n
    P: nxm matrix with known locations of transmitting beacons as columns
    d: vector in R^m where d[i] contains the noisy distance from beacon P[:, i] to x
    tol: gradient Euclidean error tolerance (stop when norm(∇f(x)) <= tol)
    max_iters: maximum iterations of Newton's method to try

Returns:
    x_trace: Vector{Vector{Float64}} containing each Newton iterate x_k in R^n. 

"""
function newton_optimizer(x0, P, d, tol, max_iters)
    return x_trace
end

"""
Use gradient descent as described in Problem 8 to solve the nonlinear optimization problem from Problem 6.
This should work for Euclidean distance measurements in any dimension n, and any number 
    of noisy measurements m.

Inputs:
    x0: initial guess for the position of the receiver in R^n
    P: nxm matrix with known locations of transmitting beacons as columns
    d: vector in R^m where d[i] contains the noisy distance from beacon P[:, i] to x
    tol: gradient Euclidean error tolerance (stop when norm(∇f(x)) <= tol)
    max_iters: maximum iterations of gradient descent to try
	gamma: step size constant

Returns:
    x_trace: Vector{Vector{Float64}} containing each gradient descent iterate x_k in R^n. 

"""
function gradient_descent(x0, P, d, tol, max_iters, gamma)
    return x_trace
end