using LinearAlgebra
using StatsBase
using Random
import PyPlot as plt
using NPZ
using ProgressMeter
using Optim 
using LaTeXStrings
using SpecialFunctions

function avgAndStd(vals)

    sz = size(vals)[1]
    avg = sum(vals)/sz
    sdev = sqrt(sum((vals .- avg).^2)/sz)

    return avg, sdev
end

function state2num(s)
    n = size(s)[1]
    return Int( dot( 2 .^((n-1):-1:0) , (s .+ 1)./2 ) )
end

function num2state(num,n)

    s = digits(num, base=2, pad=n) |> reverse

    return s.*2 .- 1
end
# s = [-1; -1; 1]
# num = state2num(s)
# display(num)
# sout = num2state(num,3)
# display(sout)

# s should be an array of +/-1's
# Note that this is the big-endian representation 
function binaryInc!(s)

    @inbounds for i=1:length(s)
        if s[i]==-1
            s[i]=1
            if i>1
                s[1:i-1] .= -1
            end
            return
        end
    end

    # If the end is reached, restart at 0
    s .= -1
end

# Determine if a state is a local minimum
# J should have zero diagonal
function isMin(s,J)
    return minimum(s .* (J*s)) > 0
end

function countMins(J)

    n = size(J)[1]
    s = -Int.(ones(n))

    cnt = 0
    for i=1:2^(n-1)
        cnt += isMin(s,J)
        binaryInc!(s)
    end

    return cnt
end

# Same as count mins except it returns the actual minima
function findMins(J)
    n = size(J)[1]
    s = -Int.(ones(n))

    cnt = 0
    mins = []
    for i=1:2^(n-1)

        isMinimum = isMin(s,J)

        if isMinimum
            cnt += 1
            push!(mins,copy(s))
        end 

        binaryInc!(s)
    end

    return cnt, mins
end

function sortUnq(vals)

    unq = unique(vals)
    numUnq = size(unq)[1]
    cnts = zeros(Int,numUnq)

    for i=1:numUnq 
        cnts[i] = sum(vals .== unq[i]) 
    end

    idx = sortperm(cnts,rev=true)
    unq = unq[idx]
    cnts = cnts[idx]

    return unq, cnts
end

# Makes a random spin state, integer type.
function randState(n)
    return rand([-1, 1], n)
end

function SK_J(n)

    J = randn(n,n) ./ sqrt(2*n)
    J += J' 
    for i=1:n 
        J[i,i] = 0
    end

    return J
end
# J = SK_J(5)
# display(J)

# The J matrix is not normalized by P.
# This allows the J matrix to be integer type, which may speed up matrix multiplication.
function Hop_J(n,P)

    patts = rand([-1;1],(P,n))
    
    J = patts[1,:]*patts[1,:]' 
    for i=2:P 
        J += patts[i,:]*patts[i,:]'
    end

    for i=1:n 
        J[i,i] = 0
    end

    return patts, J

end
# n = 10
# P = 5
# patts,J = Hop_J(n,P)
# display(patts)
# display(J)

# J should have zero diagonal
function SD!(s,J,ϵ=0)

    n = size(s)[1]

    dE = zeros(n)
    while true
        dE .= s .* (J*s)

        idx = argmin(dE)

        if dE[idx] < -ϵ
            s[idx] = -s[idx]
        else
            break
        end
    end
end
# n = 10
# J = SK_J(n)
# s = randState(n)
# E0 = -dot(s,J*s)
# SD!(s,J)
# Ef = -dot(s,J*s)
# println("E0="*string(E0))
# println("Ef="*string(Ef))

# J again needs zero on the diagonal
function SDfast!(s,J,reset=500,ϵ=0,cntMaxFactor=10)

    n = size(s)[1]

    cnt=0
    dE = zeros(n)
    while true

        # Recompute the full vector every once in a while for stability.
        if mod(cnt,reset)==0
            dE .= s .* (J*s)
        end

        idx = argmin(dE)

        if dE[idx] < -ϵ

            # Update spin flip vector
            for i=1:n 
                dE[i] -= 2*s[i]*J[i,idx]*s[idx]
            end
            dE[idx] = - dE[idx]
            
            # Update the spin itself
            s[idx] = -s[idx]
        else
            break
        end

        cnt += 1
        if cnt>=cntMaxFactor*n
            display("Did not converge!")
            return 0
        end
    end
end
# n = 50
# J = SK_J(n)
# s = randState(n)
# E0 = -dot(s,J*s)
# SDfast!(s,J)
# Ef = -dot(s,J*s)
# println("E0="*string(E0))
# println("Ef="*string(Ef))

function ZTMH!(s,J,ϵ=0)

    n = size(s)[1]
    idxs = collect(1:n)
    
    converged = false
    while converged==false

        converged = true
        
        # Try all spins in a random order.
        shuffle!(idxs)
        @inbounds for i ∈ idxs 
           
            # Compute spin flip energy without allocations or bounds checking
            dE = 0
            @inbounds for j=1:n
                if i==j
                    continue
                else
                    dE += J[i,j]*s[j]
                end
            end
            dE *= s[i]

            # Check for spin flip
            if dE < -ϵ
                s[i] = -s[i]
                converged = false
                break
            end
        end 
    end
end
# n = 10
# J = SK_J(n)
# s = rand( [-1; 1] , n)
# E0 = -dot(s,J*s)
# ZTMH!(s,J)
# Ef = -dot(s,J*s)
# println("E0="*string(E0))
# println("Ef="*string(Ef))

# ns is number of successes, nt is total
# Returns absolute value of the low and high deviations, 
# this is the convention that matplotlib errorbar likes. 
function wilsonErr(recall;z=1)

    npnts = size(recall)[1]

    # Get successes, failures, and total
    ns = recall[:,1]
    n  = recall[:,2]
    nf = n .- ns

    p0 = (ns .+ 0.5*z^2)./(n .+ z^2)

    spread = z./(n .+ z^2) .* sqrt.(ns.*nf./n .+ z^2/4)

    probs0 = recall[:,1] ./ recall[:,2]

    return abs.(p0 - spread - probs0), abs.(p0 + spread - probs0)
end
# recall = [4 1000; 10 10]
# display(recall)
# errLow, errHigh = wilsonErr(recall)
# display(errLow)
# display(errHigh)

function findCross(x,y,val)

    sz = size(x)[1]
    
    # Check boundary case
    if y[1]<val 
        return x[1]
    end

    # Find cross
    for i=2:sz 
        if y[i]<val 

            # Do linear fit
            x1 = x[i-1]
            y1 = y[i-1]
            x2 = x[i]
            y2 = y[i]
            m = (y2-y1)/(x2-x1)
            b = y1 - m*x1

            return (val-b)/m
        end
    end

    return x[end]
end

function recallCurve(mem,J,errs,reps,dynamics)

    n = size(mem)[1]
    nErrVals= size(errs)[1]

    # The first column is number of successes, second column is total trials
    recall = Int.(zeros(nErrVals,2))
    recall[:,2] .= reps

    # Allocate memory for some spins
    s = randState(n)

    # Loop through all error levels
    for e=1:nErrVals

        # Get the number of spin flip errors
        errVal = errs[e]
        for k=1:reps

            # Reset spin state and apply errors 
            s .= mem
            errLocs = sample(1:n, errVal, replace=false)
            for i ∈ errLocs
                s[i] = - s[i]
            end

            # Relax
            dynamics(s,J)

            # Did it return?
            if abs(dot(s,mem)) == n
                recall[e,1] += 1
            end
        end
    end

    return recall
end
# n = 20
# errs = 0:2:10
# reps = 50
# basinThresh = 0.5
# J = SK_J(n)
# mem = randState(n)
# SD!(mem,J)
# recall = recallCurve(mem,J,errs,reps,SD!)
# probs = recall[:,1]./recall[:,2]
# bsize = findCross(errs,probs,basinThresh)
# errLow, errHigh = wilsonErr(recall)
# plt.errorbar(errs,probs,yerr=(errLow,errHigh))
# plt.xlabel("Spin flips")
# plt.ylabel("Recall prob.")
# plt.grid(alpha=0.25)
# plt.axvline(bsize,linestyle="--",color="tab:red")
# plt.axhline(thresh,linestyle="--")
# plt.display_figs()

function autobasin(mem,J,errVals,reps,dynamics;thresh=0.5,initIndex=1)

    n = size(mem)[1]
    nErrVals= size(errs)[1]

    # The first column is number of successes, second column is total trials
    recall = Int.(zeros(nErrVals,2))

    # Dynamically changing error index
    errIdx = initIndex
    if errIdx==-1 
        errIdx = Int(floor(nErrVals/2))
    end
    errVal = errVals[errIdx]
    
    # Allocate memory for some spins
    s = randState(n)

    # Let the recall begin
    for k=1:reps

        # Reset spin state and apply errors 
        s .= mem
        errLocs = sample(1:n, errVal, replace=false)
        for i ∈ errLocs
            s[i] = - s[i]
        end

        # Relax
        dynamics(s,J)

        # Add one to the trials
        recall[errIdx,2] += 1

        # Did it return?
        returned = abs(dot(s,mem)) == n
        if returned
            recall[errIdx,1] += 1
        end

        # Update the index
        prob = recall[errIdx,1]/recall[errIdx,2]
        if prob > thresh
            errIdx += 1
        else
            errIdx -= 1
        end
        errIdx = max(1,min(nErrVals,errIdx))
        errVal = errVals[errIdx]
    end

    # Measure basin size 
    probs = recall[:,1]./recall[:,2]
    for k=1:nErrVals
        if recall[k,2]==0
            probs[k] = 1.0
        end
    end
    bsize = findCross(errVals,probs,thresh)
    
    return recall, bsize 
end
# n = 20
# errs = 0:2:Int(n/2)
# reps = 50
# J = SK_J(n)
# mem = randState(n)
# SD!(mem,J)
# thresh = 0.5
# recall, bsize = autobasin(mem,J,errs,reps,SD!;thresh=thresh,initIndex=1)
# probs = recall[:,1]./recall[:,2]
# errLow, errHigh = wilsonErr(recall)
# plt.errorbar(errs,probs,yerr=(errLow,errHigh))
# plt.xlabel("Spin flips")
# plt.ylabel("Recall prob.")
# plt.axvline(bsize,linestyle="--",color="tab:red")
# plt.grid(alpha=0.25)
# plt.axhline(thresh,linestyle="--")
# plt.ylim(-0.02,1.02)
# plt.display_figs()

# Let's now investigate some approximations used to estimate the average basin size. 
# bsize should be an integer that is smaller than n/2.
function basinVol(bsize,n)
    val = 0
    for k=0:bsize
        val += binomial(n,k)
    end
    return val
end

# Bernoulli entropy
Hfunc(x) = - x*log2(x) - (1-x)*log2(1-x)

# Approximation to basin volume for a basin of size λ*n where λ<1. 
volApprox(λ,n) = 2^(n*Hfunc(λ) - log2(n)/2) 

function volApprox2(λ,n)

    ya = -(n+1)/sqrt(2*n)
    yb = λ*n*sqrt(2/n) - (n-1)/sqrt(2*n)

    return 2^n * (0.5*erf(yb) - 0.5*erf(ya))
end

function hists(vals,bins;low=nothing,high=nothing,normed=true)

    if low==nothing 
        low = minimum(vals)
    end

    if high==nothing 
        high = maximum(vals)
    end

    if high==low 
        high += 1e-10
    end

    xvals = range(low,high,length=bins)

    idxs = Int.(round.( max.(0.0, min.(1.0, (vals.-low)./(high.-low) )) .* (bins-1) .+ 1 ))

    hist = zeros(bins)
    for i∈idxs
        hist[i] += 1
    end

    if normed
        norm = sum(hist)
        hist ./= norm 
    end

    return xvals, hist, cumsum(hist)
end

function findCrossUp(xvals,yvals,thresh)

    sz = length(xvals)

    for i=1:sz 
        if yvals[i]>=thresh
            return xvals[i]
        end
    end
    return xvals[end]
end

# Here t = exp(-α), the default corresponds to α=0.001 roughly
function MehlerFiniteSize1D(x1,x2;t=0.999,s=0.15)
    
    g = (1-2*s^2)/(1+2*s^2)

    arg  = -(1+g)*( (1+g*t^2)*(x1^2 + x2^2) - 2*(1+g)*t*x1*x2 )/(2*(1-g^2 * t^2))
    return (1+g) / (2*sqrt(1-g^2 * t^2) )* exp(arg)
end

# This is just designed to be a test of the Greens function.
# Is not optimized for speed or memory efficiency.
function evalField(x0,y0,x,y;sx=0.148,sy=0.165,t=0.999)

    szx = length(x)
    szy = length(y)

    img = zeros(szy,szx)

    xfields = complex.(zeros(szx))
    yfields = complex.(zeros(szy))
    for k=1:4

        tval = t*exp(-2π*im*(k-1)/7)

        # Separable x parts
        for i=1:szx
            xfields[i] = MehlerFiniteSize1D(x0,x[i],t=tval,s=sx)
        end
        if k>1
            xfields .*= 2
        end

        # Separable y parts
        for i=1:szy
            yfields[i] = MehlerFiniteSize1D(y0,y[i],t=tval,s=sy)
        end
        if k>1
            yfields .*= 2
        end

        for i=1:szx
            for j=1:szy
                img[j,i] += real(xfields[i]*yfields[j])
            end
        end
    end

    return img
end
# x0 = 0 
# y0 = 1.0 
# x = -8:0.025:8
# y = -8:0.025:8
# img = evalField(x0,y0,x,y)
# plt.imshow(img,interpolation="None")
# plt.display_figs()

# Fast version of 4/7 J matrix.
# x1d(y1d) is a nx(ny) dimensional array with the 1d positions for each direction. 
# Jx should be (nx,nx,4) and Jy should be (ny,ny,4), both complex array
# J should be (n,n) where n=nx*ny, real valued.
function makeJ!(x1d,y1d,Jx,Jy,J;sx=0.148,sy=0.165,t=0.999)

    nx = length(x1d)
    ny = length(y1d)
    n = nx*ny

    # Compute separable terms
    for k=1:4

        tval = t*exp(2π*im*(k-1)/7)

        # Separable x parts
        for i=1:nx
            for j=i:nx
                Jx[i,j,k] = MehlerFiniteSize1D(x1d[i],x1d[j],t=tval,s=sx)
                if k>1
                    Jx[i,j,k] *= 2
                end
                Jx[j,i,k] = Jx[i,j,k]
            end
        end

        # Separable y parts
        for i=1:ny
            for j=i:ny
                Jy[i,j,k] = MehlerFiniteSize1D(y1d[i],y1d[j],t=tval,s=sy)
                if k>1
                    Jy[i,j,k] *= 2
                end
                Jy[j,i,k] = Jy[i,j,k]
            end
        end
    end

    # Combine terms 
    for ix=1:nx
        for iy=1:ny
            ii = 1 + (ix-1) + (iy-1)*nx
            
            for jx=1:nx
                for jy=1:ny

                    jj = 1 + (jx-1) + (jy-1)*nx

                    if jj<=ii
                        continue
                    end

                    J[ii,jj]  = real(Jx[ix,jx,1]*Jy[iy,jy,1])
                    J[ii,jj] += real(Jx[ix,jx,2]*Jy[iy,jy,2])
                    J[ii,jj] += real(Jx[ix,jx,3]*Jy[iy,jy,3])
                    J[ii,jj] += real(Jx[ix,jx,4]*Jy[iy,jy,4])

                    J[jj,ii] = J[ii,jj]
                end
            end
        end
    end
end
# nx = 4
# ny = 4
# n = nx*ny
# x1d = randn(nx)*2
# y1d = randn(ny)*2
# Jx = complex.(zeros(nx,nx,4))
# Jy = complex.(zeros(ny,ny,4))
# J = zeros(n,n)
# makeJ!(x1d,y1d,Jx,Jy,J;sx=0.15,sy=0.15)
# plt.imshow(J,vmin=-2,vmax=2,cmap="RdBu")
# plt.display_figs()

# Implements COM noise
function recallCurveFluct(mem,x1d,y1d,σ,errs,reps,dynamics)

    nx = length(x1d)
    ny = length(y1d)
    n = size(mem)[1]
    nErrVals= size(errs)[1]

    # Make sure the sizes check out.
    if nx*ny != n 
        println("Sizes do not match! Exiting autobasin.")
        return nothing
    end

    # Assume isotropic fluctuations
    σx = sqrt(σ)
    σy = σx

    # Allocate memory for J matrices
    Jx = complex.(zeros(nx,nx,4))
    Jy = complex.(zeros(ny,ny,4))
    J  = zeros(n,n)

    # The first column is number of successes, second column is total trials
    recall = Int.(zeros(nErrVals,2))
    recall[:,2] .= reps

    # Allocate memory for some spins
    s = randState(n)

    # Loop through all error levels
    for e=1:nErrVals

        # Get the number of spin flip errors
        errVal = errs[e]
        for k=1:reps

            # Reset spin state and apply errors 
            s .= mem
            errLocs = sample(1:n, errVal, replace=false)
            for i ∈ errLocs
                s[i] = - s[i]
            end

            # Construct noisy J
            xfluc = x1d .+ σx*randn() 
            yfluc = y1d .+ σy*randn() 
            makeJ!(xfluc,yfluc,Jx,Jy,J)

            # Relax
            dynamics(s,J)

            # Did it return?
            if abs(dot(s,mem)) == n
                recall[e,1] += 1
            end
        end
    end

    return recall
end
# nx = 4
# ny = 5
# σ = 0.5/35
# thresh = 0.5
# errs = 0:1:Int(n/2)
# reps = 50
# n = nx*ny
# x1d = randn(nx)*2
# y1d = randn(ny)*2
# Jx = zeros(nx,nx,4)
# Jy = zeros(ny,ny,4)
# J = zeros(n,n)
# makeJ!(x1d,y1d,Jx,Jy,J)
# mem = randState(n)
# SD!(mem,J)
# # Noiseless recall
# recall = recallCurveFluct(mem,x1d,y1d,0,errs,reps,SD!)
# probs = recall[:,1]./recall[:,2]
# bsize = findCross(errs,probs,thresh)
# errLow, errHigh = wilsonErr(recall)
# plt.errorbar(errs,probs,yerr=(errLow,errHigh))
# # Noisy recall 
# recall = recallCurveFluct(mem,x1d,y1d,σ,errs,reps,SD!)
# probs = recall[:,1]./recall[:,2]
# bsize = findCross(errs,probs,thresh)
# errLow, errHigh = wilsonErr(recall)
# plt.errorbar(errs,probs,yerr=(errLow,errHigh))
# plt.xlabel("Spin flips")
# plt.ylabel("Recall prob.")
# plt.grid(alpha=0.25)
# plt.axvline(bsize,linestyle="--",color="tab:red")
# plt.axhline(thresh,linestyle="--")
# plt.display_figs()

# x1d, y1d, and σ should be in units of waists. The σ is total COM position fluctuation 
function autobasinFluct(mem,x1d,y1d,σ,errVals,reps,dynamics;thresh=0.5,initIndex=1)

    nx = length(x1d)
    ny = length(y1d)
    n = size(mem)[1]
    nErrVals= size(errs)[1]

    # Make sure the sizes check out.
    if nx*ny != n 
        println("Sizes do not match! Exiting autobasin.")
        return nothing
    end

    # Assume isotropic fluctuations
    σx = sqrt(σ)
    σy = σx

    # Allocate memory for J matrices
    Jx = complex.(zeros(nx,nx,4))
    Jy = complex.(zeros(ny,ny,4))
    J  = zeros(n,n)

    # The first column is number of successes, second column is total trials
    recall = Int.(zeros(nErrVals,2))

    # Dynamically changing error index
    errIdx = initIndex
    if errIdx==-1 
        errIdx = Int(floor(nErrVals/2))
    end
    errVal = errVals[errIdx]
    
    # Allocate memory for some spins
    s = randState(n)

    # Let the recall begin
    for k=1:reps

        # Reset spin state and apply errors 
        s .= mem
        errLocs = sample(1:n, errVal, replace=false)
        for i ∈ errLocs
            s[i] = - s[i]
        end

        # Construct noisy J
        xfluc = x1d .+ σx*randn() 
        yfluc = y1d .+ σy*randn() 
        makeJ!(xfluc,yfluc,Jx,Jy,J)

        # Relax
        dynamics(s,J)

        # Add one to the trials
        recall[errIdx,2] += 1

        # Did it return?
        returned = abs(dot(s,mem)) == n
        if returned
            recall[errIdx,1] += 1
        end

        # Update the index
        prob = recall[errIdx,1]/recall[errIdx,2]
        if prob > thresh
            errIdx += 1
        else
            errIdx -= 1
        end
        errIdx = max(1,min(nErrVals,errIdx))
        errVal = errVals[errIdx]
    end

    # Measure basin size 
    probs = recall[:,1]./recall[:,2]
    for k=1:nErrVals
        if recall[k,2]==0
            probs[k] = 1.0
        end
    end
    bsize = findCross(errVals,probs,thresh)
    
    return recall, bsize 
end
# nx = 4
# ny = 4
# σ = 0.5/35
# thresh = 0.5
# errs = 0:1:Int(n/2)
# reps = 200
# n = nx*ny
# x1d = randn(nx)*2
# y1d = randn(ny)*2
# Jx = zeros(nx,nx,4)
# Jy = zeros(ny,ny,4)
# J = zeros(n,n)
# makeJ!(x1d,y1d,Jx,Jy,J)
# mem = randState(n)
# SD!(mem,J)
# # Recall without noise
# recall,bsize = autobasinFluct(mem,x1d,y1d,0,errs,reps,SD!;thresh=thresh)
# probs = recall[:,1]./recall[:,2]
# errLow, errHigh = wilsonErr(recall)
# plt.errorbar(errs,probs,yerr=(errLow,errHigh))
# # Recall with noise
# recall,bsize = autobasinFluct(mem,x1d,y1d,σ,errs,reps,SD!;thresh=thresh)
# probs = recall[:,1]./recall[:,2]
# errLow, errHigh = wilsonErr(recall)
# plt.errorbar(errs,probs,yerr=(errLow,errHigh))
# plt.xlabel("Spin flips")
# plt.ylabel("Recall prob.")
# plt.axvline(bsize,linestyle="--",color="tab:red")
# plt.grid(alpha=0.25)
# plt.axhline(thresh,linestyle="--")
# plt.ylim(-0.02,1.02)
# plt.xlim(-0.1,n/2)
# plt.display_figs()