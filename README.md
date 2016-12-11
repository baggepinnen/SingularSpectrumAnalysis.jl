# SingularSpectrumAnalysis
A package for performing Singular Spectrum Analysis (SSA) https://en.wikipedia.org/wiki/Singular_spectrum_analysis

## Simple Usage
The plot functions are only available if `Plots.jl` has been loaded when `SingularSpectrumAnalysis` is loaded.
```julia
using SingularSpectrumAnalysis
# generate some data
L = 20 # Window length
K = 100
N = K*L; # number of datapoints
t = 1:N; # Time vector
T = 20; # period of main oscillation
y = sin(2pi/T*t); # Signal
y .+= (0.5sin(2pi/T*4*t)).^2 # Add another frequency
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

## Advanced usage
See the implementation of functions `hsvd` and `reconstruct`

## Reading
See http://www.jds-online.com/files/JDS-396.pdf for an easy-to-read introduction to SSA
