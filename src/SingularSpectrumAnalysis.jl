"""
SingularSpectrumAnalysis

Simple Usage:
```julia
# generate some data
L = 20 # Window length
K = 100
N = K*L; # number of datapoints
t = 1:N; # Time vector
T = 20; # period of main oscillation
y = sin.(2pi/T*t); # Signal
y .+= (0.5sin.(2pi/T*4*t)).^2 # Add another frequency
e = 0.1randn(N); # Add some noise
ys = y+e;
# plot(ys)

USV = hsvd(ys,L) # Perform svd on the trajectory matrix
sigmaplot(USV) # Plot normalized singular values
# logsigmaplot(USV) # Plot singular values
# cumsigmaplot(USV) # Plot cumulative normalized singular values
seasonal_groupings = [1:2, 4:5] # Determine pairs of singular values corresponding to seasonal components
trends = 3 # If some singular value lacks a buddy, this is a trend component
pairplot(USV,seasonal_groupings) # plot phase plots for all seasonal components
yrt, yrs = reconstruct(USV, trends, seasonal_groupings) # Reconstruct the underlying signal without noise, based on all identified components with significant singular values
yr = sum([yrt yrs],2) # Form full reconstruction
plot([y ys yr], lab=["y" "ys" "yr"])
```
See http://www.jds-online.com/files/JDS-396.pdf for an easy-to-read introduction to SSA
"""
module SingularSpectrumAnalysis

using LinearAlgebra

export hankel, hankelize, elementary, reconstruct, hsvd


export sigmaplot, logsigmaplot, cumsigmaplot, pairplot
import Plots
sigmaplot(USV) = Plots.scatter(USV.S./sum(USV.S), title="Normalized Singular Value Plot", ylabel="\$\\sigma / \\sum\\sigma_i\$")
logsigmaplot(USV) = Plots.scatter(USV.S, yscale=:ln, title="Singular Value Plot", ylabel="\$\\log \\sigma\$")
cumsigmaplot(USV) = Plots.scatter(cumsum(USV.S./sum(USV.S)), title="Cumulative Singular Value Plot", ylabel="\$ \\sum\\sigma_{1:i}\$")

"""
pairplot(USV, groupings)
Usage:
```julia
USV = hsvd(data,L)
seasonal_groupings = [1:2, 4:5]
pairplot(USV,seasonal_groupings)
```
"""
function pairplot(USV, groupings::AbstractArray)
    M = length(groupings)
    f = Plots.plot(layout=M)
    for m = 1:M
        i = groupings[m]
        @assert length(i) == 2 "pairplot: All groupings have to be pairs"
        elements = USV.U[:,i].*sqrt.(USV.S[i])'
        Plots.plot!(elements[:,1], elements[:,2],subplot=m, legend=false)
    end
    f
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
    USV = svdfact(X)
end

"""
    yrt,yrs = reconstruct(USV::SVD, trends, seasonal::AbstractArray)
    yr      = reconstruct(USV::SVD, groupings::AbstractArray)

Compute a reconstruction of the time-series based on an SVD object obtained from `hsvd` and user selected groupings. See also `?SingularSpectrumAnalysis`
"""
function reconstruct(USV::Base.LinAlg.SVD, trends, seasonal::AbstractArray)
    if isa(trends,Number)
        trends = [trends]
    end
    yrt = reconstruct(USV, trends)
    yrs = reconstruct(USV, seasonal)
    yrt, yrs
end

function reconstruct(USV::Base.LinAlg.SVD, groupings::AbstractArray)
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

end
