using SingularSpectrumAnalysis
using Test, Statistics, Random
const SSA = SingularSpectrumAnalysis

@testset "SingularSpectrumAnalysis" begin
    Random.seed!(1)
    # Generate some data
    L = 20
    K = 100
    N = K*L;
    t = 1:N;
    T = 20;
    y = sin.(2pi/T*t);
    y .+= (0.5sin.(2pi/T*4*t)).^2
    e = 0.1randn(N);
    yn = y+e;
    # plot(yn)

    yt, ys = analyze(yn, L)
    A,x = fit_trend(yt, 1)
    @test x ≈ [0.124, 0] atol=0.1

    USV = hsvd(yn,L)

    trend, seasonal_groupings = autogroup(USV)
    @test trend == 3
    @test seasonal_groupings == [1:2, 4:5]

    yrt, yrs = reconstruct(USV, trend, seasonal_groupings)
    yr = sum([yrt yrs],dims=2)
    @test sqrt(mean((y.-yn).^2)) > sqrt(mean((y.-yr).^2))


    using Plots
    plot(USV, cumulative=true)
    plot(USV, cumulative=false)
    pairplot(USV, seasonal_groupings)
end
@testset "forecasting" begin
    include("forecast.jl")
end
