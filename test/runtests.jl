using SingularSpectrumAnalysis
using Base.Test
const SSA = SingularSpectrumAnalysis

# Generate some data
L = 20
K = 100
N = K*L;
t = 1:N;
T = 20;
y = sin.(2pi/T*t);
y .+= (0.5sin.(2pi/T*4*t)).^2
e = 0.1randn(N);
ys = y+e;
# plot(ys)

USV = hsvd(ys,L)
seasonal_groupings = [1:2, 4:5]
trends = 3
yrt, yrs = reconstruct(USV, trends, seasonal_groupings)
yr = sum([yrt yrs],2)
@test sqrt(mean((y.-ys).^2)) > sqrt(mean((y.-yr).^2))


# using Plots
# sigmaplot(USV)
# logsigmaplot(USV)
# cumsigmaplot(USV)
