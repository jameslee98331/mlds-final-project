# Contents of this file is translated from Julia from this repository:
# https://github.com/jwmi/CoarsenedPosterior by J. W. Miller


# max number of components
MAX_COMP = 10

# Gamma shape parameter
SHAPE_ALPHA = 1 / MAX_COMP
LAMBDA = 1

# Cut-off
CUT_OFF = qgamma(1 - LAMBDA/MAX_COMP, shape = SHAPE_ALPHA, scale = 1)

# Scale for weight proposals
SIGMA = 0.25


log_v_prior <- function(v_i) {
  # Prior log density on latent weights v_i
  return(log(dgamma(v_i, shape = SHAPE_ALPHA, scale = 1)))
}


normpdf <- function(x, mu, tau) {
  # The normal probability density function, parametrised with (mean, precision)
  # where precision: tau = 1 / (sigma^2)
  return(sqrt(tau / (2 * pi)) * exp(-0.5 * tau * (x - mu) ^ 2))
}


normlogpdf <- function(x, mu, std_dev) {
  # The log-normal probability density function, parametrised with (mean, std_dev)
  return(-0.5 * log(2 * pi) - log(std_dev) - 0.5 * ((x - mu) / std_dev) ^ 2)
}


likelihood <- function(x, theta) {
  # The likelihood function of data x in normal distribution with params:
  # theta = [mean, precision]
  mean = theta[1]
  precision = theta[2]
  return(normpdf(x, mean, precision))
}

# Prior on component parameters: [mean, log(precision)]
# Params for prior on mean
m0_mean = 0
s0_mean = 5

# Params for prior on log-precision
m0_log_precision = 0
s0_log_precision = 2


log_theta_prior <- function(t) {
  return(
    normlogpdf(t[1], m0_mean, s0_mean) +
      normlogpdf(log(t[2]), m0_log_precision, s0_log_precision)
  )
}

# proposal distribution for component param moves
sm = 0.2 * s0_mean
sl = 0.2

theta_prop <- function(theta) {
  return(
    c(
      theta[1] + sm * rnorm(1),
      theta[2] * exp(rnorm(1) * sl)
    )
  )
}

log_theta_prop <- function(t, tp) {
  return(
    normlogpdf(tp[1], t[1], sm) + normlogpdf(log(tp[2]), log(t[2]), sl)
  )
}

new_thetas <- function(m) {
  # initialize params for m new components
  vals = matrix(NA, ncol = 2, nrow = m)
  for (i in 1:m) {
    vals[i, ] = c(m0_mean, exp(m0_log_precision))
  }
  return(vals)
}

sampler <- function(x, n_samples, zeta, max_comps = MAX_COMP, cut_off = CUT_OFF, sigma = SIGMA) {
  # This function runs the MH-MCMC sampler on the coarsened posterior

  # Args:
  # x = data (array of datapoints)
  # n_samples = # of MCMC iterations
  # zeta = power to raise likelihood to
  # max_comps = maximum # of mixture components
  # cut_off = cutoff point for weight map
  # sigma = scale of weight proposals

  n = length(x)

  # initialize state
  v = rep(0, max_comps)
  v[1] = 2 * cut_off
  v[-1] = cut_off / 2 # latent weights
  q = pmax(v - cut_off, 0) # mapped weights (unnormalized)
  s = sum(q)
  theta = new_thetas(max_comps)

  # initialize vars for computing likelihood
  L = matrix(NA, nrow = max_comps, ncol = n)
  for (i in 1:max_comps) {
    for (j in 1:n) {
      L[i, j] = likelihood(x[j], theta[i, ])
    }
  }
  M = q %*% L # mixture density with unnormalized weights
  ll = sum(log(M)) - n * log(s)  # log-lik

  # Mp and Lp will hold proposed values for Mixture Density and Likelihood
  Mp = rep(0, n)
  Lp = rep(0, n)

  # Arrays to keep track of value at each iteration
  k_r = rep(0, n_samples)
  m_r = matrix(0, nrow = n_samples, ncol = max_comps)
  l_r = matrix(0, nrow = n_samples, ncol = max_comps)
  v_r = matrix(0, nrow = n_samples, ncol = max_comps)

  # number of theta proposals accepted
  num_theta_props_accepted = 0

  # number of v proposals accepted
  num_v_props_accepted = 0

  for (iter in 1:n_samples) {
    # update parameters thetas with Metropolis-Hastings moves
    for (i in 1:max_comps) {
      thetap = theta_prop(theta[i, ])
      llp = -n * log(s)

      for (j in 1:n) {
        Lp[j] = likelihood(x[j], thetap)

        # Note: max(., 0) prevents negative values due to roundoff error.
        Mp[j] = max(M[j] + q[i] * (Lp[j] - L[i, j]), 0)

        llp = llp + log(Mp[j])
      }

      # compute acceptance probability
      top = log_theta_prior(thetap) + (zeta * llp) + log_theta_prop(thetap, theta[i, ])
      bot = log_theta_prior(theta[i, ]) + (zeta * ll) + log_theta_prop(theta[i, ], thetap)
      p_accept = min(1, exp(top - bot))

      # accept or reject
      if (runif(1) < p_accept) {
        theta[i, ] = thetap

        for (j in 1:n) {
          L[i, j] = Lp[j]
        }

        # Accept Proposals
        Mp_temp = Mp
        M_temp = M
        M = Mp_temp
        Mp = M_temp
        ll = llp

        # Increment num_theta_props_accepted if accepted
        num_theta_props_accepted = num_theta_props_accepted + 1
      }
    }

    # update weights with Metropolis-Hastings moves
    for (i in 1:max_comps) {

      vp = v[i] * exp(rnorm(1) * sigma)
      qp = max(vp - cut_off, 0)
      sp = s + qp - q[i]

      if (sp > 0) {
        llp = -n * log(sp)

        for (j in 1:n) {
          Mp[j] = max(M[j] + (qp - q[i]) * L[i, j], 0)
          llp = llp + log(Mp[j])
        }

        # compute acceptance probability
        top = log_v_prior(vp) + (zeta * llp) - log(v[i])
        bot = log_v_prior(v[i]) + (zeta * ll) - log(vp)
        p_accept = min(1, exp(top - bot))

        # accept or reject
        if (runif(1) < p_accept) {

          # Accept the candidate
          v[i] = vp
          q[i] = qp
          s = sp
          Mp_temp = Mp
          M_temp = M
          M = Mp_temp
          Mp = M_temp
          ll = llp

          # Increment num_v_props_accepted if accepted
          num_v_props_accepted = num_v_props_accepted + 1
        }
      }
    }

    # record
    k_r[iter] = sum(q > 0)  # number of active components
    for (i in 1:max_comps) {
      m_r[iter, i] = theta[i, 1]  # means
      l_r[iter, i] = theta[i, 2]  # precisions
      v_r[iter, i] = v[i]  # latent weights
    }
  }


  # MH acceptance rate for theta proposals
  acceptance_rate_theta_props = num_theta_props_accepted / (max_comps * n_samples)

  # MH acceptance rate for v proposals
  acceptance_rate_v_props = num_v_props_accepted / (max_comps * n_samples)

  output = list(
    "p" = (q / s),
    "theta" = theta,
    "k_r" = k_r,
    "v_r" = v_r,
    "art" = acceptance_rate_theta_props,
    "arv" = acceptance_rate_v_props,
    "m_r" = m_r,
    "l_r" = l_r
  )
  return(output)
}

skewrnd = function(n, loc, scale, shape) {
  # This function generates n skew normal data points
  # n - number of points to generate
  # loc - location parameter
  # scale - scale parameter
  # shape - shape parameter

  data = rep(NA, n)
  for (i in 1:n) {
    z = rnorm(1)
    if (shape * z < rnorm(1)) {
      data[i] = loc - scale * z
    } else {
      data[i] = loc + scale * z
    }
  }
  return(data)
}

