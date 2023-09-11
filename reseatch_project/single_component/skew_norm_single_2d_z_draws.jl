using Distributed
addprocs(3)
@everywhere using BayesianMixtures
@everywhere using HDF5
@everywhere using JLD

set = 8
alpha = 1
ns = [100, 500, 1000]

@everywhere function run_simulation(x, mcmc_its, mcmc_burn, t_max)
    mfm_options = BayesianMixtures.options(
        "MVN",
        "MFM",
        x,
        mcmc_its,
        n_burn=mcmc_burn,
        t_max=t_max
    )
    # Results of Miller and Harrison (2018)
    mfm_result = BayesianMixtures.run_sampler(mfm_options)
    return mfm_result
end

@sync @distributed for n in ns
    all_data = h5read(
        "./data_inputs/skew_norm/2d/single_skew_normal_2d-alpha=$alpha-set-$set.jld",
        "data"
    )
    data = [all_data[j, :]::Array{Float64} for j in 1:n]
    mcmc_its = 100000
    mcmc_burn = 50000
    t_max = 100
    result = run_simulation(data, mcmc_its, mcmc_burn, t_max)
    save(
        "./comp_outputs/skew_norm/2d/z_draws-single_skew_normal_2d-alpha=$alpha-n=$n-set-$set.jld",
        "z",
        result.z
    )
end
