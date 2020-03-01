module SingularSpectrumAnalysis

using LinearAlgebra, Statistics, RecipesBase, Requires, TotalLeastSquares

export hankel, hankelize, unhankel, elementary, reconstruct, hsvd, autogroup, analyze, rpca, lowrankfilter, esprit

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
const hankelize = unhankel

"""
    USV = hsvd(y,L;robust=false)

Form a trajectory hankel matrix from data `y` and compute svd on this Hankelmatrix
"""
function hsvd(y,L;robust=false)
    X = hankel(y,L) # Form trajectory matrix
    if robust
        X = rpca(X)[1]
    end
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
        yr[:,m] = unhankel(X)
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
    iseven(ind) && (ind += 1)
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


"""
    analyze(y, L; robust=true)

Automatically analyze the given signal. `robust` indicates whether or not to use a robust decomposition resistant to outliers.

#Arguments:
- `y`: Signal
- `L`: Lag embedding dimension (window size)
"""
function analyze(y,L; robust=true)
    USV = hsvd(y,L, robust=robust)
    trend, seasonal_groupings = autogroup(USV)
    reconstruct(USV, trend, seasonal_groupings)
end

"""
    esprit(x, L, r; fs=1, robust=false)

Estimate `r` (positive) frequencies present in signal `x` using a lag-correlation matrix of size `L`.

R. Roy and T. Kailath, "ESPRIT-estimation of signal parameters via rotational invariance techniques," in IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 37, no. 7, pp. 984-995, Jul 1989.

#Arguments:
- `x`: Signal
- `L`: Size of lag embedding and the covariance matrix used.
- `r`: Number of frequencies, in contrast to `DSP.esprit`, we return `r` positive frequencies whereas `DSP.esprit` return `r÷2` positive/negative frequency pairs.
- `fs`: Sample rate
- `robust`: Whether or not to use a robust decomposition resistant to outliers.
"""
function esprit(x::AbstractArray{T}, L, r; fs=T(1), robust=false) where T
    N = length(x)
    USV = hsvd(x,L,robust=robust)
    D = eigvals( USV.U[1:end-1,1:2r] \ USV.U[2:end,1:2r] )
    sort(fs/(2π) .* filter(x->x>0 , angle.(D)))
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
        models = [CSI.ar(1, ys[:,i], ar_order, estimator=CSI.tls)[1] for i = 1:ns] # Fit one model per component
        # models = [CSI.ar(1, ys[:,i], ar_order)[1] for i = 1:ns] # Fit one model per component
        # We can use these models to for one-step predictions
        seasonal_predictions = [CSI.predict(models[i], ys[:,i])  for i = 1:ns]
        for s in seasonal_predictions
            for i = 1:ar_order insert!(s,1,0.) end
        end
        PredictionData(x,A,models,seasonal_predictions)
    end

    function _pred(pd::PredictionData)
        A,x,ys,models = pd.trend_regressor, pd.trend_parameters, pd.seasonal_predictions, pd.seasonal_models
        na = length(CSI.denvec(models[1])[])
        @assert size(A,1) >= 3
        @assert (ys isa AbstractVecOrMat{<:Number} && size(ys,1) >= 3) || size(ys[1],1) >= 3
        ns = length(models)
        ysh = [CSI.predict(models[i], reverse(ys[i][end-na:end]))  for i = 1:ns]
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
        for i = eachindex(ys)
            push!(ys[i], ysh[i])
        end
        pd.trend_regressor = [A; (size(A,1)+1).^(0:trendorder(pd))']
        pred(pd,n-1)
    end
end
end
end
