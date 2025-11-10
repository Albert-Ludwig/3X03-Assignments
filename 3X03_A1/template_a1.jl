using LinearAlgebra

"""
Implement the function signatures defined here in your solution file <FIRSTNAME>_LASTNAME_a1.jl.
DO NOT MODIFY THE SIGNATURES.

Use test_a1.jl to test your solutions.
"""


"""
Perform backward substitution to solve the system U*x = b for x.
Inputs:
    U: an nxn upper diagonal square matrix (assume the diagonal is nonzero)
    b: an nx1 vector
Output: 
    x: the solution to U*x = b
"""
function backward_substitution(U, b)
    return x
end

"""
Perform forward substitution to solve the system L*x = b for x.
Inputs:
    L: an nxn lower diagonal square matrix (assume the diagonal is nonzero)
    b: an nx1 vector
Output: 
    x: the solution to L*x = b
"""
function forward_substitution(L, b)
    return x
end

"""
Gaussian elimination without pivoting. 
You may assume A_in has full rank and never has a zero pivot. 

Input:
    A: a full rank nxn matrix with no zero pivots.
    b: a nx1 vector

Output:
    A_out: the input matrix A in row-echelon form
    b_out: the input vector b after having undergone the transformations putting A into row-echelon form
"""
function gaussian_elimination(A, b)
    return A_out, b_out
end

"""
LU decomposition with partial pivoting. 

Input: 
    A_in: an nxn full-rank matrix

Returns:
    L: nxn lower triangular matrix
    U: nxn upper triangular matrix
    p: permuted vector of indices 1:n representing the pivots 
       (i.e., your solution should (approximately) satisfy A[p, :] = L*U) 
"""
function lu_partial_pivoting(A)
    return L, U, p
end

"""
Solve A*x = b for x given the LU decomposition (approximately) satisfying A[p, :] = L*U.

Inputs:
    L: lower triangular matrix
    U: upper triangular matrix
    p: vector of permuted indices representing the pivots

Output:
    x: solution to A*x = b (recall that A[p, :] = L*U)
"""
function lu_solve(L, U, p, b)
    return x
end
