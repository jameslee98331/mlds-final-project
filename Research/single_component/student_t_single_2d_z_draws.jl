using Distributed
addprocs(5)
@everywhere using BayesianMixtures
@everywhere using HDF5
@everywhere using JLD

set = 8
dof = 50
ns = [100, 250, 500, 750, 1000]

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
        "./data_inputs/student_t/2d/single_t_2d-dof=$dof-set-$set.jld",
        "data"
    )
    data = [all_data[j, :]::Array{Float64} for j in 1:n]
    mcmc_its = 10^5
    mcmc_burn = 5 * 10^4
    t_max = 250
    result = run_simulation(data, mcmc_its, mcmc_burn, t_max)
    save(
        "./comp_outputs/student_t/2d/z_draws-single_t_2d-dof=$dof-n=$n-set-$set.jld",
        "z",
        result.z
    )
end
