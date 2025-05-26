using LinearAlgebra

# Test matrix with negative eigenvalue, triangle inequality, and row/column sums of 1
M = [1 -1 -1; -1 1 -1; -1 -1 1]

# Check eigenvalues
eigenvals = eigvals(M)
println("Eigenvalues: ", eigenvals)

# Verify row and column sums
println("\nRow sums: ", sum(M, dims=2))
println("Column sums: ", sum(M, dims=1))

# Verify triangle inequality
function verify_triangle_inequality_matrix(M)
    n = size(M, 1)
    for i in 1:n
        for j in 1:n
            for k in 1:n
                if M[i, j] > M[i, k] + M[k, j]
                    println("Triangle inequality violated for i=$i, j=$j, k=$k")
                    println("M[$i,$j] = $(M[i,j]) > $(M[i,k]) + $(M[k,j]) = M[$i,$k] + M[$k,$j]")
                    return false
                end
            end
        end
    end
    println("\nTriangle inequality satisfied for all values!")
    return true
end

verify_triangle_inequality_matrix(M)

function verify_reverse_triangle_inequality_matrix(M)
    n = size(M, 1)
    for i in 1:n
        for j in 1:n
            for k in 1:n
                if 1 + M[i, j] < M[i, k] + M[k, j]
                    println("Reverse Triangle inequality violated for i=$i, j=$j, k=$k")
                    println("M[$i,$j] = $(M[i,j]) < $(M[i,k]) + $(M[k,j]) = M[$i,$k] + M[$k,$j]")
                    return false
                end
            end
        end
    end
    println("\nReverse Triangle inequality satisfied for all values!")
    return true
end

verify_reverse_triangle_inequality_matrix(M)