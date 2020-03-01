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
    y .+= (0.5sin.(2pi/T*4*t))
    y .+ 0.01 .*(1:N)
    e = 0.1randn(N);
    yn = y+e;
    # plot(yn)

    yt, ys = analyze(yn, L, robust=false)
    A,x = fit_trend(yt, 1)
    @test √sum(abs2,x) < 2e-3

    USV = hsvd(yn,L)

    trend, seasonal_groupings = autogroup(USV)
    @test trend == 5
    @test seasonal_groupings == [1:2, 3:4]

    yrt, yrs = reconstruct(USV, trend, seasonal_groupings)
    yr = sum([yrt yrs],dims=2)
    @test sqrt(mean((y.-yn).^2)) > sqrt(mean((y.-yr).^2))


    using Plots
    plot(USV, cumulative=true)
    plot(USV, cumulative=false)
    pairplot(USV, seasonal_groupings)


    USV = hsvd(yn,L,robust=true)

    trend, seasonal_groupings = autogroup(USV)
    @test trend == 5
    @test seasonal_groupings == [1:2, 3:4]


    USV = hsvd(yn,2L,robust=true)

    trend, seasonal_groupings = autogroup(USV)
    @test trend == 5
    @test seasonal_groupings == [1:2, 3:4]


    @testset "With trend" begin
        @info "Testing With trend"


        y = sin.(2pi/T*t);
        y .+= (0.5sin.(2pi/T*4*t))
        y .+ 0.01 .*(1:N)

        y .+= 0.001 .* (1:N)
        e = 0.1randn(N);
        yn = y+e;
        # plot(yn)

        yt, ys = analyze(yn, L, robust=false)
        A,x = fit_trend(yt, 1)
        @test x[2] ≈ 0.001 rtol = 1e-3

        USV = hsvd(yn,L)

        trend, seasonal_groupings = autogroup(USV)
        @test trend == 1
        @test seasonal_groupings == [2:3, 4:5]

        yrt, yrs = reconstruct(USV, trend, seasonal_groupings)
        yr = sum([yrt yrs],dims=2)
        @test sqrt(mean((y.-yn).^2)) > sqrt(mean((y.-yr).^2))


        yn[end÷2] = 10000 # introduce outlier

        yt, ys = analyze(yn, L, robust=true)
        A,x = fit_trend(yt, 1)
        @test x[2] ≈ 0.001 rtol = 1e-1

        USV = hsvd(yn,L,robust=true)

        trend, seasonal_groupings = autogroup(USV)
        @test trend == 1
        @test seasonal_groupings == [2:3, 4:5]

        yrt, yrs = reconstruct(USV, trend, seasonal_groupings)
        yr = sum([yrt yrs],dims=2)
        @test sqrt(mean((y.-yn).^2)) > sqrt(mean((y.-yr).^2))

    end


    @testset "ESPRIT" begin
        @info "Testing ESPRIT"
        T = 10
        t = 1:10000
        s = repeat([ones(T÷2);-ones(T÷2)],length(t)÷T) .* sin.(LinRange(0,200pi,length(t))) .|> Float32
        freqs = SingularSpectrumAnalysis.esprit(s,200T,5)
        @test freqs ≈ [0.5, 0.31, 0.28, 0.11, 0.09] atol=0.015

    end

    @testset "forecasting" begin
        include("forecast.jl")
    end
end
