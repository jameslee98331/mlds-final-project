# Map-MCMC for mixtures.

# This code assumes the following functions have been defined:
#   likelihood(x[j], theta)
#   log_v_prior(v)
#   log_theta_prior(theta)
#   theta_prop(theta)
#   log_theta_prop(theta,thetap)
#   new_thetas(m)

function sampler(data, mcmc_its, t_max, cutoff, sigma, zeta)
    # data = data (array of datapoints)
    # mcmc_its = # of MCMC iterations
    # t_max = maximum # of mixture components
    # cutoff = cutoff point for weight map
    # sigma = scale of weight proposals
    # zeta = power to raise likelihood to

    num_samples = length(data)

    # Initialize states
    # weight map
    v = zeros(t_max)
    v[1] = 2 * cutoff
    v[2:end] .= cutoff / 2 # latent weights
    q = max.(v .- cutoff, 0) # mapped weights (unnormalized)
    s = sum(q)

    theta = new_thetas(t_max)

    # initialize vars for computing likelihood
    L = [likelihood(data[j], theta[i]) for i = 1:t_max, j = 1:num_samples]

    # mixture density with unnormalized weights
    M = vec(q' * L)

    # log-lik
    loglik = sum(log.(M)) - num_samples * log(s)

    # Mp and Lp will hold proposed values
    M_proposed = zeros(num_samples)
    L_proposed = zeros(num_samples)

    # record-keeping
    k_r = zeros(mcmc_its)
    m_r = zeros(mcmc_its, t_max)
    l_r = zeros(mcmc_its, t_max)
    v_r = zeros(mcmc_its, t_max)
    num_theta_accepted = 0  # number of theta proposals accepted
    num_v_accepted = 0  # number of v proposals accepted

    # draw samples
    for iter = 1:mcmc_its

        # update parameters with Metropolis-Hastings moves
        for i = 1:t_max
            theta_proposed = theta_prop(theta[i])
            loglik_proposed = -num_samples * log(s)

            for j = 1:num_samples
                L_proposed[j] = likelihood(data[j], theta_proposed)

                # Note: max(.,0) prevents negative values due to roundoff error.
                M_proposed[j] = max(M[j] + q[i] * (L_proposed[j] - L[i, j]), 0)
                loglik_proposed += log(M_proposed[j])
            end

            # compute acceptance probability
            top = log_theta_prior(theta_proposed) + zeta * loglik_proposed + log_theta_prop(theta_proposed, theta[i])
            bot = log_theta_prior(theta[i]) + zeta * loglik + log_theta_prop(theta[i], theta_proposed)
            p_accept = min(1, exp(top - bot))

            # accept or reject
            if rand() < p_accept
                theta[i] = copy(theta_proposed)

                for j = 1:num_samples
                    L[i, j] = L_proposed[j]
                end

                M, M_proposed = M_proposed, M
                loglik = loglik_proposed
                num_theta_accepted += 1
            end
        end

        # update weights with Metropolis-Hastings moves
        for i = 1:t_max
            vp = v[i] * exp(randn() * sigma)
            qp = max(vp - c, 0)
            sp = s + qp - q[i]

            if sp > 0
                loglik_proposed = -num_samples * log(sp)

                for j = 1:num_samples
                    M_proposed[j] = max(M[j] + (qp - q[i]) * L[i, j], 0)
                    loglik_proposed += log(M_proposed[j])
                end

                # compute acceptance probability
                top = log_v_prior(vp) + zeta * loglik_proposed - log(v[i])
                bot = log_v_prior(v[i]) + zeta * loglik - log(vp)
                p_accept = min(1, exp(top - bot))

                # accept or reject
                if rand() < p_accept
                    v[i] = vp
                    q[i] = qp
                    s = sp
                    # swap
                    M, M_proposed = M_proposed, M
                    loglik = loglik_proposed
                    num_v_accepted += 1
                end
            end
        end # for i = 1:t_max

        # record
        k_r[iter] = sum(q .> 0) # number of active components
        for i = 1:t_max
            m_r[iter, i] = theta[i][1]  # means
            l_r[iter, i] = theta[i][2]  # precisions
            v_r[iter, i] = v[i]  # latent weights
        end # for i = 1:t_max
    end # for i = 1:mcmc_its

    acceptance_rate_theta = num_theta_accepted / (t_max * mcmc_its)  # MH acceptance rate for theta proposals
    acceptance_rate_v = num_v_accepted / (t_max * mcmc_its)  # MH acceptance rate for v proposals
    p = (q / s)

    return p, theta, k_r, v_r, z_r, acceptance_rate_theta, acceptance_rate_v, m_r, l_r
end
