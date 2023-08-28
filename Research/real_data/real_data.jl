using Distributed
addprocs(5)
@everywhere using BayesianMixtures
@everywhere using HDF5
@everywhere using JLD

ns = [100, 200, 300, 400, 500, 600, 700, 800, 900]

@everywhere function run_simulation(x, mcmc_its, mcmc_burn, t_max)
    mfm_options = BayesianMixtures.options(
        "MVN",
        "MFM",
        x,
        mcmc_its,
        n_burn=mcmc_burn,
        t_max=t_max
    )
    mfm_result = BayesianMixtures.run_sampler(mfm_options)
    return mfm_result
end

@sync @distributed for n in ns
    all_data = h5read(
        "./data_inputs/pyramidal CA1.jld",
        "data"
    )
    data = [all_data[:, j]::Array{Float64} for j in 1:n]
    mcmc_its = 10^5
    mcmc_burn = 5 * 10^4
    t_max = n
    result = run_simulation(data, mcmc_its, mcmc_burn, t_max)
    k_posterior = BayesianMixtures.k_posterior(result)
    save(
        "./comp_outputs/k_posteriors-pyramidal_ca1-n=$n.jld",
        "k_posteriors",
        k_posterior
    )
    save(
        "./comp_outputs/z_draws-pyramidal_ca1-n=$n.jld",
        "z_draws",
        result.z
    )
    save(
        "./comp_outputs/t_draws-pyramidal_ca1-n=$n.jld",
        "t",
        result.t
    )
end
