using BayesianMixtures
using HDF5
using JLD

n = 100
alpha = 7
set = 1

function run_simulation(x, mcmc_its, mcmc_burn, t_max)
    mfm_options = BayesianMixtures.options(
        "Normal",
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

all_data = h5read(
    "../data_inputs/skew_norm/1d/single_skew_normal_1d-alpha=$alpha-set-$set.jld",
    "data"
)

# create dataset with 1 single component
data = [all_data[j]::Float64 for j in 1:n]

# run MFM sampler
mcmc_its = 10^5
mcmc_burn = Int(mcmc_its / 10)
t_max = 150
result = run_simulation(data, mcmc_its, mcmc_burn, t_max)

t_r = result.t
save(
    "./ts.jld",
    "t_r",
    t_r
)

BayesianMixtures.plot_autocorrelation(result, 1)
BayesianMixtures.plot_t_running(result)
