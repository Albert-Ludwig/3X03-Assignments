#Problem 1
function newton_int(x, y)
    n = length(x)
    c = copy(y) # newton interplolation coefficients, initial c0=y
    for j = 2:n
        for i = n:-1:j
            c[i] = (c[i] - c[i-1]) / (x[i] - x[i-j+1]) #compute the divided differences
        end
    end
    return c
end

#Problem 2
function horner(c, x, X)
    n = length(c)
    m = length(X)
    p = zeros(m)
    for k = 1:m
        value = c[n] # initial with cn
        for i = n-1:-1:1
            value = value * (X[k] - x[i]) + c[i] # evaluate using Horner's method, use the nested structure.
        end
        p[k] = value
    end
    return p
end

#Problem 3
function composite_trapezoidal_rule(f, a, b, r)
    h = (b - a) / r # step size
    sum_val = f(a) + f(b) # upper base + lower base of trapezoid
    for i = 1:(r-1)
        sum_val += 2 * f(a + i * h) # use the iteration to sum the whole interval
    end
    return (h / 2) * sum_val
end

function composite_midpoint_rule(f, a, b, r)
    h = (b - a) / r # step size
    sum_val = 0.0
    for i = 1:r
        midpoint = a + (i - 0.5) * h # midpoint of each subinterval
        sum_val += f(midpoint) # sum the function values at midpoints
    end
    return h * sum_val # the whole rectangle area
end

function composite_simpsons_rule(f, a, b, r)
    # note the r is even
    h = (b - a) / r # step size
    sum_val = f(a) + f(b)
    for i = 1:2:(r-1) # first half interval
        sum_val += 4 * f(a + i * h)
    end
    for i = 2:2:(r-2) # second half interval
        sum_val += 2 * f(a + i * h)
    end
    return (h / 3) * sum_val
end