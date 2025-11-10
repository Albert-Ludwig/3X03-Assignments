using LinearAlgebra
#Johnson Ji 400499564 assignment 1

# question 3
"""
Gaussian elimination with partial pivoting
Transforms matrix A to upper triangular form and applies same operations to vector b
Returns: (U, b_modified) where U is upper triangular
"""
function gaussian_elimination(A, b)
    n = size(A, 1)                                    # Get matrix dimension
    bvec = vec(b)                                     # Convert b to vector
    T = promote_type(eltype(A), eltype(bvec), Float64) # Determine common numeric type
    A_out = Matrix{T}(A)                              # Create working copy of A
    b_out = Vector{T}(bvec)                           # Create working copy of b

    @inbounds for k in 1:n-1                         # Forward elimination
        pivot = A_out[k, k]                           # Current pivot element
        for i in k+1:n                                # Eliminate below pivot
            m = A_out[i, k] / pivot                   # Multiplier for elimination
            if m != 0
                A_out[i, k] = zero(T)                 # Zero out below pivot
                for j in k+1:n                        # Update rest of row
                    A_out[i, j] -= m * A_out[k, j]
                end
                b_out[i] -= m * b_out[k]              # Apply same operation to b
            end
        end
    end
    return A_out, b_out
end

# question 4
"""
Backward substitution for upper triangular system Ux = b
Solves from bottom to top: x[n], x[n-1], ..., x[1]
"""
function backward_substitution(U, b)
    n = size(U, 1)                                    # Get system size
    bvec = vec(b)                                     # Convert b to vector
    T = promote_type(eltype(U), eltype(bvec), Float64) # Determine common type
    x = Vector{T}(undef, n)                           # Initialize solution vector
    @inbounds for i in n:-1:1                        # Solve backwards
        s = zero(T)                                   # Initialize sum
        for j in i+1:n                                # Sum known terms
            s += U[i, j] * x[j]
        end
        x[i] = (bvec[i] - s) / U[i, i]               # Solve for x[i]
    end
    return x
end

"""
Forward substitution for lower triangular system Lx = b
Solves from top to bottom: x[1], x[2], ..., x[n]
"""
function forward_substitution(L, b)
    n = size(L, 1)                                    # Get system size
    bvec = vec(b)                                     # Convert b to vector
    T = promote_type(eltype(L), eltype(bvec), Float64) # Determine common type
    x = Vector{T}(undef, n)                           # Initialize solution vector
    @inbounds for i in 1:n                           # Solve forwards
        s = zero(T)                                   # Initialize sum
        for j in 1:i-1                                # Sum known terms
            s += L[i, j] * x[j]
        end
        x[i] = (bvec[i] - s) / L[i, i]               # Solve for x[i]
    end
    return x
end

# question 5
"""
LU decomposition with partial pivoting
Decomposes A into L*U = P*A where:
- L is lower triangular with unit diagonal
- U is upper triangular
- P is permutation matrix
"""
function lu_partial_pivoting(A)
    n = size(A, 1)                                    # Get matrix dimension
    T = promote_type(eltype(A), Float64)              # Determine numeric type
    U = Matrix{T}(A)                                  # Working copy for U
    L = Matrix{T}(I, n, n)                            # Initialize L as identity
    P = Matrix{T}(I, n, n)                            # Initialize P as identity

    @inbounds for k in 1:n-1                         # For each column
        # Find row with largest absolute value in column k
        pivot_row = argmax(abs.(U[k:n, k])) + k - 1

        # Swap rows if needed for partial pivoting
        if pivot_row != k
            U[[k, pivot_row], k:n] = U[[pivot_row, k], k:n]     # Swap rows in U
            P[[k, pivot_row], :] = P[[pivot_row, k], :]         # Update permutation
            if k > 1
                L[[k, pivot_row], 1:k-1] = L[[pivot_row, k], 1:k-1] # Swap in L
            end
        end

        # Gaussian elimination step
        for i in k+1:n
            L[i, k] = U[i, k] / U[k, k]              # Store multiplier in L
            if L[i, k] != 0
                U[i, k] = zero(T)                     # Zero out below pivot
                @inbounds @simd for j in k+1:n       # Update rest of row
                    U[i, j] -= L[i, k] * U[k, j]
                end
            end
        end
    end

    return L, U, P
end

# question 6
"""
Solve linear system using LU decomposition
Given L, U, P from LU decomposition, solve Ax = b
Steps: 1) Apply permutation: Pb
       2) Forward substitution: Ly = Pb
       3) Backward substitution: Ux = y
"""
function lu_solve(L, U, P, b)
    bvec = vec(b)                                     # Convert b to vector
    rhs = P * bvec                                    # Apply permutation
    y = forward_substitution(L, rhs)                  # Solve Ly = Pb
    x = backward_substitution(U, y)                   # Solve Ux = y
    return x
end