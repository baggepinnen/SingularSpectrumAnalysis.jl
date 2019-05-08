using SingularSpectrumAnalysis, Test
L = 20
K = 10
N = K*L;
t = 1:N;
T = 20;
y = sin.(2pi/T*t);            # Add seasons
y .+= (0.5sin.(2pi/T*4*t)).^2 # Add seasons
y .+= LinRange(0,1,N)         # Add trend
e = 0.1randn(N);
yn = y+e;                     # Add noise

yt, ys = analyze(yn, L) # trend and seasons
using ControlSystemIdentification
pd  = PredictionData(yt,ys, trend_order=1, ar_order=2)
yth = trend(pd)
ysh = seasons(pd)
@test yth isa Vector{Float64}
@test ysh isa Vector{<:Vector}
@test length(yth) == N
@test length(ysh[1]) == N-2
@test length(ysh) >= 1
plot(pd)

pd  = pred(pd,2) # pd now contains extended fields with the predictions in the end
plot(pd)
yth = trend(pd)
ysh = seasons(pd)

@test yth isa Vector{Float64}
@test ysh isa Vector{<:Vector}
@test length(yth) == N+2
@test length(ysh[1]) == N

@test trend(pd,2) == trend(pd)[2]
@test seasons(pd,2) == getindex.(seasons(pd),2)
