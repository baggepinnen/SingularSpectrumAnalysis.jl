module SingularSpectrumAnalysis

using LinearAlgebra, Statistics, RecipesBase

export hankel, hankelize, elementary, reconstruct, hsvd, autogroup, analyze, fit_trend

export sigmaplot, pairplot

@recipe function sigmaplot(USV::SVD; cumulative=false)
    seriestype := :scatter
    title --> "Normalized Singular Value Plot"
    ylabel --> "\$\\sigma / \\sum\\sigma_i\$"
    S = USV.S./sum(USV.S)
    cumulative ? cumsum(S) : S
end



@userplot Pairplot
"""
pairplot(USV, groupings)
Usage:
```julia
USV = hsvd(data,L)
seasonal_groupings = [1:2, 4:5]
pairplot(USV,seasonal_groupings)
```
"""
pairplot

@recipe function pairplot(h::Pairplot)
    USV, groupings = h.args[1:2]
    M = length(groupings)
    layout := M
    legend --> false
    for m = 1:M
        i = groupings[m]
        @assert length(i) == 2 "pairplot: All groupings have to be pairs"
        elements = USV.U[:,i].*sqrt.(USV.S[i])'
        subplot := m
        @series elements[:,1], elements[:,2]
    end
    nothing
end



"""
    X = hankel(x,window_size)
Form the trajectory matrix `X` of size KxL, K = N-L+1
x can be a vector or a matrix for multivariate SSA
"""
function hankel(x,L)
    N = size(x,1)
    D = isa(x,AbstractVector) ? 1 : size(x,2)
    @assert L <= N/2 "L has to be less than N/2 = $(N/2)"
    K = N-L+1
    X = zeros(K,L*D)
    for d = 1:D
        for j = 1:L, i = 1:K
            k = i+j-1
            X[i,j+(d-1)*L] = x[k,d]
        end
    end
    X
end

"""
    Ui = elementary(USV,I)
Computes the sum Uᵢ*Sᵢ*Vᵢ' for all i in I
If I is 1:L, the returned matrix is identical to U*S*V'
"""
function elementary(USV,I)
    sum(USV.U[:,i]*USV.S[i]*USV.Vt[i,:]' for i in I)
end

"""
    means = hankelize(X)
Computes a timeseries from an approximate Hankel matrix by diagoal averaging (Hankelization). Note that the returned value is a vector and not a hankel matrix. A hankel matrix is easily obtainable by `hankel(hankelize(X), size(X,2))`
"""
function hankelize(X)
    K,L = size(X)
    @assert K >= L "The input matrix must be a tall matrix"
    N = K+L-1
    means = zeros(N)
    for n = 1:N
        if n <= K
            rangei = n:-1:max(1,n-L+1)
            rangej = 1:min(n,L)
        else
            rangei = K:-1:max(n-L+1,n-K+2)
            rangej = n-K+1:L
        end
        means[n] = mean(X[rangei[k],rangej[k]] for k in eachindex(rangei))
    end
    means
end
"""
    USV = hsvd(y,L)
Form a trajectory hankel matrix from data `y` and compute svd on this Hankelmatrix
"""
function hsvd(y,L)
    X = hankel(y,L) # Form trajectory matrix
    USV = svd(X)
end

"""
    yrt,yrs = reconstruct(USV::SVD, trends, seasonal::AbstractArray)
    yr      = reconstruct(USV::SVD, groupings::AbstractArray)

Compute a reconstruction of the time-series based on an SVD object obtained from `hsvd` and user selected groupings. See also `?SingularSpectrumAnalysis`
"""
function reconstruct(USV, trends, seasonal::AbstractArray)
    if isa(trends,Number)
        trends = [trends]
    end
    yrt = reconstruct(USV, trends)
    yrs = reconstruct(USV, seasonal)
    size(yrt,2) == 1 && (yrt = vec(yrt))
    yrt, yrs
end

function reconstruct(USV, groupings::AbstractArray)
    M = length(groupings)
    K,L = size(USV.U)
    N = K+L-1
    yr = zeros(N,M)
    for m = 1:M
        X = elementary(USV,groupings[m])
        yr[:,m] = hankelize(X)
    end
    yr
end

"""
trend, seasonal_groupings = autogroup(USV, th = 0.95)
Try to automatically group singular values. `th ∈ (0,1)` determins the percentage of variance to explain.
"""
function autogroup(USV, th = 0.95)
    nS = USV.S .- minimum(USV.S)
    nS ./= sum(nS)
    cs = cumsum(nS)
    ind = findfirst(x->x > th, cs)
    iseven(ind) && (ind -= 1)
    t = findtrend(nS[1:ind])
    seasonal_groupings = UnitRange[]
    for i = [1:2:t-1; t+1:2:ind]
        push!(seasonal_groupings, i:i+1)
    end
    t, seasonal_groupings
end

function findtrend(S)
    function eval_trend(ind)
        iseven(ind) && return Inf
        cost = 0.0
        for i = [1:2:ind-1; ind+1:2:length(S)]
            cost += S[i] - S[i+1]
        end
        cost
    end
    costs = eval_trend.(1:length(S))
    _,ind = findmin(costs)
    ind
end

function analyze(y,L)
    USV = hsvd(y,L)
    trend, seasonal_groupings = autogroup(USV)
    reconstruct(USV, trend, seasonal_groupings)
end

"""
A,x = fit_trend(yt::AbstractVector, order=1)
Fit an n:th order polynomial to the trend signal `yt`
Returns the regressor matrix `A` and the coefficients `x` with the first coefficient being the 0:th order component (mean).
"""
function fit_trend(yt::AbstractVector, order=1)
    N = length(yt)
    A = (1:N).^(0:order)'
    x = A\yt
    A,x
end
end
