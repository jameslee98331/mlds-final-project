using Distributed
addprocs(10)
@everywhere using Distributions
@everywhere using HDF5
@everywhere using JLD
@everywhere using Random
@everywhere using Dates

@everywhere include("setup_1d.jl")

ns = [5000, 7500, 10000]::Array{Int}
n_sets = 50
sets = 1:n_sets
alphas = [10, 50, 70, 80, 100, 1000]::Array{Int}

@everywhere function histogram(x, edges=[]; n_bins=50, weights=ones(length(x)))
    if isempty(edges)
        mn, mx = minimum(x), maximum(x)
        r = mx - mn
        edges = range(mn - r / n_bins, stop=mx + r / n_bins, length=n_bins)
    else
        n_bins = length(edges) - 1
    end

    counts = zeros(Float64, n_bins)
    for i in eachindex(x)
        for j = 1:n_bins
            if (edges[j] < x[i] <= edges[j+1])
                counts[j] += weights[i]
                break
            end
        end
    end
    return counts, edges
end

@everywhere function run_simulation(data, mcmc_its, mcmc_burn, t_max, c, sigma, zeta)
    elapsed_time = (
        @elapsed p, theta, k_r, v_r, art, arv, m_r, s_r = sampler(
        data, mcmc_its, t_max, c, sigma, zeta
    )
    )
    time_per_step = elapsed_time / mcmc_its
    println("Elapsed time = $elapsed_time seconds")
    println("Time per step = $time_per_step seconds")

    # Compute posterior on k
    use = mcmc_burn+1:mcmc_its

    # bins are: (0,1],(1,2],...,(t_max-1,t_max]
    counts, edges = histogram(k_r[use], 0:t_max)
    k_posterior = counts / length(use)

    return k_posterior
end

@sync @distributed for set in sets
    all_data = h5read(
        "../single_component/data_inputs/skew_norm/1d/single_skew_normal_1d-alpha=7-set-$set.jld",
        "data"
    )

    for alpha in alphas
        for n in ns

            data = [all_data[j]::Float64 for j in 1:n]

            zeta = (1 / n) / ((1 / n) + (1 / alpha))

            println(Dates.now())
            println("n = $n, set = $set, alpha = $alpha, zeta=$zeta")

            # Run sampler
            mcmc_its = 10^5
            mcmc_burn = Int(mcmc_its / 10)
            t_max = 10

            k_posterior, z_r = run_simulation(
                data, mcmc_its, mcmc_burn, t_max, c, sigma, zeta
            )

            save_fullpath = "./comp_outputs/skew_norm/1d/k_posterior-single_skew_normal_1d-coarsen=$alpha-alpha=7-n=$n-set-$set.jld"
            save(save_fullpath, "k_posterior", k_posterior)
            # save_fullpath = "./comp_outputs/skew_norm/1d/z_draws-single_skew_normal_1d-coarsen=$alpha-alpha=7-n=$n-set-$set.jld"
            # save(save_fullpath, "z_draws", z_draws)
            println(Dates.now())
        end
    end
end
