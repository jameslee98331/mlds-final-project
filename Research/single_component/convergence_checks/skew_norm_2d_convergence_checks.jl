using BayesianMixtures
using HDF5
using JLD

ns = [100, 250, 500, 750, 1000, 1500, 2500, 5000, 7500, 10000]
alpha = 7
set = 8

function run_simulation(x, mcmc_its, mcmc_burn, t_max)
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

all_data = h5read(
    "../data_inputs/skew_norm/2d/single_skew_normal_2d-alpha=$alpha-set-$set.jld",
    "data"
)

for n in ns
    data = [all_data[j, :]::Array{Float64} for j in 1:n]

    # run MFM sampler
    mcmc_its = 10^5
    mcmc_burn = 5 * 10^4
    t_max = 100
    result = run_simulation(data, mcmc_its, mcmc_burn, t_max)
    save(
        "./t_draws-single_skew_normal_2d-alpha=$alpha-n=$n.jld",
        "t_draws",
        result.t
    )
end
