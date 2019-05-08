module SingularSpectrumAnalysis

using LinearAlgebra, Statistics, RecipesBase, Requires

export hankel, hankelize, elementary, reconstruct, hsvd, autogroup, analyze

export sigmaplot, pairplot

export PredictionData, trendorder, fit_trend, pred, trend, seasons

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
x can be a vector or a matrix For multivariate SSA
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

mutable struct PredictionData
    trend_parameters
    trend_regressor
    seasonal_models
    seasonal_predictions
end
trendorder(pd) = length(pd.trend_parameters)-1

trend(pd::PredictionData) = pd.trend_regressor*pd.trend_parameters
trend(pd::PredictionData, i) = pd.trend_regressor[i,:]'*pd.trend_parameters
seasons(pd::PredictionData) = pd.seasonal_predictions
seasons(pd::PredictionData,i) = getindex.(pd.seasonal_predictions,i)
Base.getindex(pd::PredictionData) = trend(pd,i), seasons(pd,i)

@recipe function plotpd(pd::PredictionData)
    layout := 1+length(pd.seasonal_models)
    legend --> false
    @series begin
        subplot := 1
        title --> "Estimated trend"
        pd.trend_regressor*pd.trend_parameters
    end
    for (i,s) in enumerate(pd.seasonal_predictions)
        @series begin
            subplot := 1+i
            title --> "Seasonal predictions $i"
            s
        end
    end
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

function __init__()
    @require ControlSystemIdentification="3abffc1c-5106-53b7-b354-a47bfc086282" begin
    CSI = ControlSystemIdentification

    function PredictionData(yt,ys; trend_order=1, ar_order=2)
        A,x = fit_trend(yt, trend_order)  # Fits a polynomial of order 1 (line)
        ns = size(ys,2) # number of seasonal components
        models = [CSI.ar(1, ys[:,i], ar_order)[1] for i = 1:ns] # Fit one model per component
        # We can use these models to for one-step predictions
        seasonal_predictions = [CSI.predict(models[i], ys[:,i])  for i = 1:ns]
        # for s in seasonal_predictions
        #     for i = 1:ar_order insert!(s,1,0.) end
        # end
        PredictionData(x,A,models,seasonal_predictions)
    end

    function _pred(pd::PredictionData)
        A,x,ys,models = pd.trend_regressor, pd.trend_parameters, pd.seasonal_predictions, pd.seasonal_models
        @assert size(A,1) >= 3
        @assert (ys isa AbstractVecOrMat{<:Number} && size(ys,1) >= 3) || size(ys[1],1) >= 3
        ns = length(models)
        ysh = [CSI.predict(models[i], ys[i][end-2:end])  for i = 1:ns]
        ysh = reduce(hcat, ysh)
        yth = A[end,:]'*x
        yth, ysh
    end


    """
    pd = pred(pd::PredictionData, n=1)
    Form `n` step prediction. This requires you to have constructed a `PredictionData` object according to
    ```julia
    yt, ys = analyze(yn, L) # trend and seasons
    pd = fit_trend(yt, nt)  # Fits a polynomial of order nt (nt = 1 -> line)
    na = 2 # Autoregressive order
    # Next line fits one AR model for each seasonal component
    ns = size(ys,2) # number of seasonal components
    using ControlSystemIdentification
    models = [ar(1, ys[:,i], na)[1] for i = 1:ns] # Fit one model per component
    pd.seasonal_models = models
    seasonal_predictions = [predict(models[i], ys[:,i])  for i = 1:ns]
    pd.seasonal_predictions = seasonal_predictions
    ```
    """
    function pred(pd::PredictionData, n=1)
        n == 0 && (return pd)
        A,x,ys,models = pd.trend_regressor, pd.trend_parameters, pd.seasonal_predictions, pd.seasonal_models
        yth, ysh = _pred(pd)
        pd.trend_regressor = [A; (size(A,1)+1).^(0:trendorder(pd))']
        for i = eachindex(ys)
            pd.seasonal_predictions[i] = [ys[i]; ysh[i]]
        end
        pred(pd,n-1)
    end
end
end
end
