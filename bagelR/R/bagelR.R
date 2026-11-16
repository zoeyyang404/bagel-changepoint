setClass("bagel_type")

.bagel_pruning <- function(pruning)
{
  match.arg(
    pruning,
    c("bagel", "LP-M", "LP-P")
  )
}

#' Create a sequential changepoint detector
#'
#' To detect the changepoint sequentially, we create a detector and 
#' then pass observations to [update()] one at a time.
#' 
#' @inheritParams bagel_kv_exact
#' @param feature_vector Function `feature_vector(t, tau)` returning a numeric
#'   column matrix, which is the same as shown in
#'   [bagel_kv_exact()].
#' @param max_particles The maximum number of particles, at least three, including
#'   no change. 
#' @details
#' The function provides the sequential version of the bagel detectors in [bagel_kv_exact()], [bagel_kv_approximate()] and [bagel_uv_approximate()].
#' In the sequential settings, we need to create a detector object first and then update the detector with new observations.
#' We apply the stopping threshold in our observation loop; the detector itself
#' continues updating until we stop supplying observations.
#' 
#' The detector can be created using [sequential_bagel_kv_exact()], [sequential_bagel_kv_approximate()] or
#' [sequential_bagel_uv_approximate()] for change-in-mean with known variance, other settings with known variance and other settings with unknown variance.
#'
#' [update()] changes the detector directly and returns a no-change weight.
#' We retain the original detector for subsequent updates and record change
#' probabilities as `1 - no_change`. After at least one update, we can inspect
#' the posterior with [weights()] and [active_posteriors()].
#'
#' @returns An S4 detector of class `bagel_kv_exact_type`,
#'   `bagel_kv_approximate_type`, or `bagel_uv_approximate_type`,
#'   matching the constructor name.
#' @seealso [bagel_kv_exact()], [update()], [active_posteriors()]
#' @examples
#' feature <- function(t, tau) matrix(if (tau == 0) 1 else c(1, 1), ncol = 1)
#' prior <- function(t) {
#'   if (t == 1) list(mu = matrix(0), sigma = matrix(1))
#'   else list(mu = matrix(0, 2, 1), sigma = matrix(c(1, -1, -1, 2), 2))
#' }
#' transform <- function(tau) matrix(if (tau == 0) 1 else c(1, 1), nrow = 1)
#' detector <- sequential_bagel_kv_exact(
#'   feature, prior, transform, p0 = 0.9, p = 1, sigma = 1, max_particles = 10)
#' change_probability <- numeric(0)
#' for (y in c(0, 0.1, -0.1, 3, 3.1)) {
#'   no_change <- bagelR::update(detector, y)
#'   change_probability <- c(change_probability, 1 - no_change)
#'   if (1 - no_change > 0.95) break
#' }
#' change_probability
#' bagelR::time(detector)
#' bagelR::weights(detector)
#' active_posteriors(detector)
#' @aliases sequential-bagel
#' @rdname sequential-bagel
sequential_bagel_kv_exact <- function(feature_vector,prior,transform,p0,p,sigma,max_particles,pruning="bagel")
{
   pruning <- .bagel_pruning(pruning)
   bagel_object <- new("bagel_kv_exact_type",feature_vector=feature_vector,
				             prior=prior,
			                     transform=transform,
			                     p0=p0,
			                     p=p,
			                     sigma=sigma,
			                     max_particles=max_particles,
			                     bagel_object=new(bagelR_kv_exact,p0,p,sigma,max_particles,pruning))
  return(bagel_object)

}


#' @rdname sequential-bagel
sequential_bagel_kv_approximate <- function(feature_vector,prior,transform,p0,p,sigma,max_particles,pruning="bagel")
{
   pruning <- .bagel_pruning(pruning)
   bagel_object <- new("bagel_kv_approximate_type",feature_vector=feature_vector,
				             prior=prior,
			                     transform=transform,
			                     p0=p0,
			                     p=p,
			                     sigma=sigma,
			                     max_particles=max_particles,
			                     bagel_object=new(bagelR_kv_approximate,p0,p,sigma,max_particles,pruning))
  return(bagel_object)

}

#' @rdname sequential-bagel
sequential_bagel_uv_approximate <- function(feature_vector,prior,transform,p0,p,nu,iota,max_particles,pruning="bagel")
{
   pruning <- .bagel_pruning(pruning)
   bagel_object <- new("bagel_uv_approximate_type",feature_vector=feature_vector,
						   prior=prior,
			                  	   transform=transform,
			                  	   p0=p0,
			                  	   p=p,
			                  	   nu=nu,
			                  	   iota=iota,
			                  	   max_particles=max_particles,
                                                   bagel_object=new(bagelR_uv_approximate,p0,p,nu,iota,max_particles,pruning))
  return(bagel_object)				   
}




setGeneric("set_weights",function(object,weights) standardGeneric("set_weights"))
setMethod("set_weights","bagel_type",
          function(object,weights)
	  {
         object@bagel_object$set_weights(weights)     
	  })


#' Read probabilities and posterior estimates
#'
#' `1 - weights_tau_zero(fit)` provides the probability of having the change,
#'  [weights()] provides the recovered posterior probabilities of the change locations, 
#'  and [active_weights()] provides the posterior probabilities of the active groups. We can also read the posterior means, covariance factors,
#' and coefficient estimates with [active_posteriors()]. These functions
#' read a fitted result or a sequential detector after an update.
#'
#' @param object A fitted result from [bagel_kv_exact()] or a related fitting
#'   function, or a sequential detector after at least one observation.
#'   `mus()` and `sigmas()` accept sequential detectors only; `nu()` and
#'   `iota()` additionally require an unknown-variance detector.
#' @param ... Additional arguments to the `weights` generic. No additional
#'   arguments are used by the bagelR methods.
#' @details
#' A location `tau = k` means that the change starts at observation `k + 1`;
#' `tau = 0` means no change. With `pruning = "bagel"` or `"LP-M"`, the algorithm uses the merging rules to combine neighbouring candidates into groups. 
#' `taus()` gives each group's right endpoint, and `ratios()` gives the relative weights of
#' the locations within it. `weights()` combines the group weight with
#' these ratios to recover a weight for each location.
#'
#' With `pruning = "LP-P"`, candidates are removed instead of merged.
#' Removed locations are absent from `weights()`, so use `taus()` to find
#' which candidates remain. Weights are not renormalised immediately after
#' removal and may therefore sum to less than one.
#'
#' For a sequential detector, `weights_tau_zero()` is the current scalar
#' no-change weight. For a fitted result it gives the history of these weights,
#' one for each processed observation. This history is shorter than the input
#' if the analysis stopped early. The first no-change weight is `p0`.
#'
#' @returns
#' * `weights()`: numeric vector of recovered location weights, beginning
#'   with no change. Under the merging strategies, entries correspond to
#'   `tau = 0, ..., T - 1`, where `T = bagelR::time(object)`.
#' * `active_weights()`: numeric vector of active group weights, beginning
#'   with no change, with at most `max_particles(object)` entries when
#'   the configured limit is at least three.
#' * `active_posteriors()`: list of named lists, one per active group, with
#'   `tau`, `weight`, coefficient mean column matrix `mu`, and
#'   covariance-factor matrix `sigma`. Unknown-variance models also have
#'   inverse-gamma shape `nu` and scale `iota`. Conditional coefficient
#'   covariance is the observation variance times `sigma`.
#' * `taus()`: numeric vector of active representative changepoint locations.
#' * `ratios()`: list of numeric vectors giving within-group relative weights.
#'   An all-zero group can have all-zero ratios.
#' * `weights_tau_zero()`: scalar for a detector, or numeric history for a
#'   fitted result, on the ordinary probability scale. Very small values
#'   can underflow to zero; fitted results also retain log weights.
#' * `time()`: numeric scalar giving the number of processed observations.
#' * `max_particles()`: numeric scalar giving the configured particle limit.
#' @section Coefficient and variance parameters:
#' The `sigma` matrices returned by [active_posteriors()] are covariance
#' factors. Multiplying by observation variance gives conditional coefficient
#' covariance. For known variance, the multiplier is the constructor's
#' `sigma^2`. Unknown-variance models use inverse-gamma shape `nu` and scale
#' `iota`, with density proportional to
#' \eqn{v^{-(\nu + 1)} \exp(-\iota/v)} at variance \eqn{v}.
#'
#' For sequential detectors, `mus()` and `sigmas()` also return lists of
#' mean column matrices and covariance factors, in active-particle order.
#' `nu()` and `iota()` each return a numeric scalar for the no-change particle
#' only. We use [active_posteriors()] when we need every particle's parameters
#' or are inspecting a fitted result.
#' @seealso [bagel_kv_exact()], [sequential_bagel_kv_exact()], [changepoint()]
#' @aliases bagel-accessors weights
#' @rdname bagel-accessors
setGeneric("weights",function(object) standardGeneric("weights"))
#' @rdname bagel-accessors
#' @usage NULL
setMethod("weights","bagel_type",
          function(object)
	  {

	    return(Reduce(c,Map("*",object@bagel_object$get_weights(),object@bagel_object$get_ratios())))

	  })


#' @rdname bagel-accessors
setGeneric("active_weights",function(object) standardGeneric("active_weights"))
#' @rdname bagel-accessors
#' @usage NULL
setMethod("active_weights","bagel_type",
          function(object)
	  {
        return(object@bagel_object$get_weights())
	  })

#' @rdname bagel-accessors
setGeneric("active_posteriors",function(object) standardGeneric("active_posteriors"))
#' @rdname bagel-accessors
#' @usage NULL
setMethod("active_posteriors","bagel_type",
          function(object)
	  {
	    return(object@bagel_object$get_posteriors())
	  })


#' @rdname bagel-accessors
setGeneric("taus",function(object) standardGeneric("taus"))
#' @rdname bagel-accessors
#' @usage NULL
setMethod("taus","bagel_type",
          function(object)
	  {
	    return(object@bagel_object$get_taus())
	  })



#' @rdname bagel-accessors
setGeneric("ratios",function(object) standardGeneric("ratios"))
#' @rdname bagel-accessors
#' @usage NULL
setMethod("ratios","bagel_type",
          function(object)
	  {
	    return(object@bagel_object$get_ratios())	
	  })


#' @rdname bagel-accessors
setGeneric("weights_tau_zero",function(object) standardGeneric("weights_tau_zero"))
#' @rdname bagel-accessors
#' @usage NULL
setMethod("weights_tau_zero","bagel_type",
          function(object)
	  {
	    return(object@bagel_object$get_weights()[1])
	  })



#' @rdname bagel-accessors
setGeneric("time",function(object) standardGeneric("time"))
#' @rdname bagel-accessors
#' @usage NULL
setMethod("time","bagel_type",
          function(object)
	  {
	    return(object@bagel_object$get_time() - 1)
	  })

#' @rdname bagel-accessors
setGeneric("max_particles",function(object) standardGeneric("max_particles"))
#' @rdname bagel-accessors
#' @usage NULL
setMethod("max_particles","bagel_type",
          function(object)
	  {
	    return(object@bagel_object$get_max_num_particles())
	  })


#' @aliases bagel-posterior
#' @rdname bagel-accessors
setGeneric("mus",function(object) standardGeneric("mus"))
#' @rdname bagel-accessors
#' @usage NULL
setMethod("mus","bagel_type",
          function(object)
	  {
	    return(object@bagel_object$get_mus())
	  })

#' @rdname bagel-accessors
setGeneric("sigmas",function(object) standardGeneric("sigmas"))
#' @rdname bagel-accessors
#' @usage NULL
setMethod("sigmas","bagel_type",
          function(object)
	  {
	    return(object@bagel_object$get_sigmas())
	  })

setClass("bagel_result_type", slots=list(y = "numeric",
                                         threshold = "numeric",
					 max_particles = "numeric",
                                         w0ts = "numeric",
                                         log_w0ts = "numeric",
					 weights="numeric",
					 posteriors="list",
					 ratios="list",
					 taus = "numeric",
					 t="numeric",
					 trace="list"))

bagel_result <- function(y,
                         threshold,
			 max_particles,
                         w0ts,
			 weights,
			 posteriors,
			 ratios,
			 taus,
			 t,
			 trace,
                         log_w0ts = log(w0ts))
{
	return(new("bagel_result_type",y=y,
	                               threshold=threshold,
				       max_particles=max_particles,
	                               w0ts=w0ts,
                                       log_w0ts=log_w0ts,
				       weights=weights,
				       posteriors=posteriors,
				       ratios=ratios,
				       taus=taus,
				       t=t,
				       trace=trace))
}


setClass("bagel_uv_approximate_type", slots=list(feature_vector = "function",
      		       			         prior = "function",
				                 transform = "function",
				                 p0 = "numeric",
				                 p = "numeric",
				                 nu = "numeric",
	                                         iota = "numeric",	
				                 max_particles = "numeric",
				                 bagel_object = "ANY"),
						 contains="bagel_type")

#' @rdname bagel
bagel_uv_approximate <- function(Y,feature_vector,prior,transform,p0,p,nu,iota,max_particles,threshold,pruning="bagel")
{
   pruning <- .bagel_pruning(pruning)
   bagel_object <- new("bagel_uv_approximate_type",feature_vector=feature_vector,
						   prior=prior,
			                  	   transform=transform,
			                  	   p0=p0,
			                  	   p=p,
			                  	   nu=nu,
			                  	   iota=iota,
			                  	   max_particles=max_particles,
                                                   bagel_object=new(bagelR_uv_approximate,p0,p,nu,iota,max_particles,pruning))
  return(analyse(Y,threshold,bagel_object))

}

setClass("bagel_uv_exact_type", slots=list(feature_vector = "function",
      		       			   prior = "function",
				           transform = "function",
				           p0 = "numeric",
				           p = "numeric",
				           nu = "numeric",
	                                   iota = "numeric",	
				           max_particles = "numeric",
				           bagel_object = "ANY"),
					   contains="bagel_type")

bagel_uv_exact <- function(Y,feature_vector,prior,transform,p0,p,nu,iota,max_particles,threshold,pruning="bagel")
{
   pruning <- .bagel_pruning(pruning)
   bagel_object <- new("bagel_uv_exact_type",feature_vector=feature_vector,
				             prior=prior,
			                     transform=transform,
			                     p0=p0,
			                     p=p,
			                     nu=nu,
			                     iota=iota,
			                     max_particles=max_particles,
			                     bagel_object=new(bagelR_uv_exact,p0,p,nu,iota,max_particles,pruning))
  return(analyse(Y,threshold,bagel_object))
}



setClass("bagel_kv_approximate_type", slots=list(feature_vector = "function",
      		       			         prior = "function",
				                 transform = "function",
				                 p0 = "numeric",
				                 p = "numeric",
				                 sigma = "numeric",
				                 max_particles = "numeric",
				                 bagel_object = "ANY"),
						 contains="bagel_type")

#' @rdname bagel
bagel_kv_approximate <- function(Y,feature_vector,prior,transform,p0,p,sigma,max_particles,threshold,pruning="bagel")
{
   pruning <- .bagel_pruning(pruning)

   bagel_object <- new("bagel_kv_approximate_type",feature_vector=feature_vector,
						   prior=prior,
			                  	   transform=transform,
			                  	   p0=p0,
			                  	   p=p,
			                  	   sigma=sigma,
			                  	   max_particles=max_particles,
                                                   bagel_object=new(bagelR_kv_approximate,p0,p,sigma,max_particles,pruning))
  return(analyse(Y,threshold,bagel_object))

}


setClass("bagel_kv_exact_type", slots=list(feature_vector = "function",
      		       			   prior = "function",
				           transform = "function",
				           p0 = "numeric",
				           p = "numeric",
				           sigma = "numeric",
				           max_particles = "numeric",
					   bagel_object = "ANY"),
					   contains="bagel_type")

#' @rdname bagel-accessors
setGeneric("nu",function(object) standardGeneric("nu"))
#' @rdname bagel-accessors
#' @usage NULL
setMethod("nu","bagel_uv_exact_type",
          function(object)
	  {
	    return(object@bagel_object$get_nus()[[1]])
	  })
#' @rdname bagel-accessors
#' @usage NULL
setMethod("nu","bagel_uv_approximate_type",
          function(object)
	  {
	    return(object@bagel_object$get_nus()[[1]])
	  })

#' @rdname bagel-accessors
setGeneric("iota",function(object) standardGeneric("iota"))
#' @rdname bagel-accessors
#' @usage NULL
setMethod("iota","bagel_uv_exact_type",
          function(object)
	  {
	    return(object@bagel_object$get_iotas()[[1]])
	  })
#' @rdname bagel-accessors
#' @usage NULL
setMethod("iota","bagel_uv_approximate_type",
          function(object)
	  {
	    return(object@bagel_object$get_iotas()[[1]])
	  })

#' Fit a single-changepoint linear model
#'
#' The function fits a linear model with at most one changepoint on a univariate time series. After each observation, update
#' the probability of no change and the probabilities of possible change
#' locations. The analysis stops when the change probability exceeds the
#' `threshold`. Set `threshold = 1` or `Inf` to process the whole series.
#'
#' @param Y Nonempty numeric vector of finite observations, in time order.
#' @param feature_vector Function `feature_vector(t, tau)` returning a
#'   numeric column matrix. Here `t` is the one-based observation index,
#'   `tau = 0` denotes no change, and `tau = k > 0` denotes a change
#'   beginning at observation `k + 1`. See Details for dimensions.
#' @param prior Function `prior(t)` returning a list with numeric column
#'   matrix `mu` and symmetric positive-definite matrix `sigma`.
#'   `prior(1)` specifies the no-change coefficients; subsequent calls
#'   specify the joint pre-change and change coefficients. `sigma` is a
#'   covariance factor, not an observation standard deviation.
#' @param transform Function `transform(tau)` returning a numeric matrix
#'   that maps the candidate's coefficients to the post-change
#'   coefficients compared during merging.
#' @param p0 Prior probability of no change, with `0 < p0 <= 1`.
#' @param p Location-prior parameter. This argument is kept for compatibility,
#'   but the code currently uses a uniform prior over change locations
#'   regardless of its value. Use `p = 1`.
#' @param sigma Positive known observation-noise standard deviation for
#'   the `kv` variants. The observation variance is `sigma^2`.
#' @param nu Positive shape of the inverse-gamma prior on observation
#'   variance for the `uv` variants.
#' @param iota Positive scale of the inverse-gamma prior on observation
#'   variance for the `uv` variants. Its density at variance `v` is
#'   proportional to \eqn{v^{-(\nu + 1)} \exp(-\iota/v)}.
#' @param max_particles Integer particle limit, at least three, including
#'   the no-change particle. Use at least `length(Y)` to avoid pruning.
#' @param threshold Stopping threshold for the posterior change probability.
#'   With a value in `[0, 1)`, stop at the first observation whose change
#'   probability is strictly above this value. Use `1` or greater, including
#'   `Inf`, to process all observations.
#' @param pruning One of `"bagel"` (posterior-distance-based merging),
#'   `"LP-M"` (merge the group with lowest posterior probability into its right
#'   neighbour), or `"LP-P"` (remove the change candidate with lowest posterior probability).
#'   The no-change particle is retained by all strategies.
#' @details
#' Each run models at most one changepoint. Before a change, the observation
#' mean is a linear function of the no-change coefficients. A change adds
#' another set of coefficients and features. If there are
#' `d` no-change coefficients and `q` change coefficients,
#' `feature_vector(t, 0)` and `prior(1)$mu` have `d` rows;
#' `feature_vector(t, tau)` for positive `tau` and `prior(t)$mu` for
#' `t > 1` have `d + q` rows. All mean and feature matrices have one column.
#' Each prior `sigma` is square with matching dimension. `transform(tau)`
#' must have matching columns and the same output dimension for the
#' candidates being compared. The covariance of the coefficients,
#' conditional on observation variance `v`, is `v * prior(t)$sigma`.
#'
#' The names `exact` and `approximate` refer to the distance used to choose
#' which particles to merge. `bagel_kv_exact()` uses exact Gaussian total variation distance
#' when the pre-change coefficients are one dimensional and variance is known.
#' For other settings, we use `bagel_kv_approximate()` or `bagel_uv_approximate()`
#' to select merges using an upper bound on the total variation distance.
#'
#' With `"LP-P"`, deleted locations are not recovered and weights are
#' not renormalised immediately after deletion; see [weights()].
#' @returns A fitted result of class `bagel_result_type`. We read the result
#'   with the following functions:
#'   * [changepoint()]: estimated change location and first alarm time.
#'   * `1 - weights_tau_zero(fit)`: change probability at each processed step.
#'   * [weights()]: final weights over possible change locations.
#'   * [active_posteriors()]: final coefficient and variance parameters.
#'   * [time()]: number of processed observations.
#'
#'   The result also retains the full input in `fit@y`, the threshold,
#'   log no-change weights, and active group weights and locations.
#'   Posterior quantities describe the final processed step;
#'   the probability history can be shorter than the input after an alarm.
#' @seealso [sequential_bagel_kv_exact()], [changepoint()], [active_weights()]
#' @examples
#' feature <- function(t, tau) matrix(if (tau == 0) 1 else c(1, 1), ncol = 1)
#' prior <- function(t) {
#'   if (t == 1) list(mu = matrix(0), sigma = matrix(1))
#'   else list(mu = matrix(0, 2, 1), sigma = matrix(c(1, -1, -1, 2), 2))
#' }
#' transform <- function(tau) matrix(if (tau == 0) 1 else c(1, 1), nrow = 1)
#' y <- c(0, 0.1, -0.1, 0.2, 3, 3.1, 2.9, 3.2)
#' fit <- bagel_kv_exact(
#'   y, feature, prior, transform, p0 = 0.9, p = 1, sigma = 1,
#'   max_particles = 5, threshold = Inf)
#' changepoint(fit)
#' weights(fit)
#' 1 - weights_tau_zero(fit)
#' active_posteriors(fit)
#'
#' unknown <- bagel_uv_approximate(
#'   y, feature, prior, transform, p0 = 0.9, p = 1, nu = 3, iota = 2,
#'   max_particles = 5, threshold = 0.95)
#' changepoint(unknown)
#' 1 - weights_tau_zero(unknown)
#' @aliases bagel
#' @rdname bagel
bagel_kv_exact <- function(Y,feature_vector,prior,transform,p0,p,sigma,max_particles,threshold,pruning="bagel")
{
   pruning <- .bagel_pruning(pruning)

   bagel_object <- new("bagel_kv_exact_type",feature_vector=feature_vector,
				             prior=prior,
			                     transform=transform,
			                     p0=p0,
			                     p=p,
			                     sigma=sigma,
			                     max_particles=max_particles,
			                     bagel_object=new(bagelR_kv_exact,p0,p,sigma,max_particles,pruning))
  return(analyse(Y,threshold,bagel_object))

}




analyse <- function(Y,threshold,bagel_object)
{
   log_cutoff <- if (threshold >= 1) Inf else -log1p(-threshold)
   log_w0ts <- list()
   for(y in Y)
   {
      update(bagel_object,y)
      log_w0t <- bagel_object@bagel_object$get_log_w0()
      log_w0ts <- append(log_w0ts,log_w0t)
      if(-log_w0t > log_cutoff)
      {
	break
      }
   }
   results <- bagel_result(as.numeric(Y),
                       threshold,
		       bagel_object@bagel_object$get_max_num_particles(),
                       exp(as.numeric(log_w0ts)),
                       as.numeric(bagel_object@bagel_object$get_weights()),
		       bagel_object@bagel_object$get_posteriors(),
		       bagel_object@bagel_object$get_ratios(),
		       as.numeric(bagel_object@bagel_object$get_taus()),	
                       bagel_object@bagel_object$get_time(),
		       list(),
                       log_w0ts=as.numeric(log_w0ts))
  return(results)
	
}












#' Update a sequential changepoint detector
#'
#' We use `update()` to incorporate one new observation into an existing
#' detector. This updates the changepoint weights and the posterior
#' distributions of the model parameters.
#'
#' @param object A detector created by [sequential_bagel_kv_exact()],
#'   [sequential_bagel_kv_approximate()], or [sequential_bagel_uv_approximate()].
#' @param y A single finite numeric observation, supplied in time order.
#' @details
#' We create the detector once, then call `bagelR::update(detector, y)`
#' for each new observation. Each call changes the same detector, which
#' we retain for subsequent observations.
#'
#' After an update, we inspect the quantities stored in `detector` with:
#'
#' * `weights(detector)`: the weights associated with the changepoint locations.
#' * `active_weights(detector)`: the weights of the retained groups.
#' * `active_posteriors(detector)`: the coefficient posterior means and
#'   covariance factors for each retained group, together with the variance
#'   parameters for an unknown-variance model.
#'
#' See [sequential_bagel_kv_exact()] for a complete example and
#' [active_posteriors()] for the definitions of the posterior quantities.
#' @returns The current no-change weight, returned as a single number.
#'   In `w0 <- bagelR::update(detector, y)`, `w0` receives this weight,
#'   and `1 - w0` gives the probability of a change after incorporating `y`.
#'
#'   The updated model remains in `detector`. We keep this object for
#'   subsequent updates and use the functions above to read its results.
#' @seealso [sequential_bagel_kv_exact()], [weights()], [active_weights()],
#'   [active_posteriors()], [weights_tau_zero()], [time()]
update <- function(object,y)
{
        bagel_object <- object@bagel_object 
	feature_vector <- object@feature_vector
	prior <- object@prior
	transform <- object@transform
	taus <- bagel_object$get_taus()
	t <- bagel_object$get_time()
	taus <- c(taus,t-1)
	bagel_object$set_feature_vectors(taus,Map(function(tau) return(feature_vector(t,tau)),taus))
	ts <- c(1,taus[-1],t)
	priors <- Map(prior,ts)
	prior_mus <- Map(function(x) return(x$mu), priors)
	prior_sigmas <- Map(function(x) x$sigma, priors)
	bagel_object$set_priors(ts,prior_mus,prior_sigmas)
	bagel_object$set_transformations(taus,Map(function(tau) return(transform(tau)),taus))
        result <- bagel_object$update(y)
	return(result)
}


#' @rdname bagel-accessors
#' @usage NULL
setMethod("weights","bagel_result_type",
          function(object)
	  {
	    return(Reduce(c,Map("*",object@weights,object@ratios)))
	  })


#' @rdname bagel-accessors
#' @usage NULL
setMethod("active_weights","bagel_result_type",
          function(object)
	  {
	    return(object@weights)
	  })

#' @rdname bagel-accessors
#' @usage NULL
setMethod("active_posteriors","bagel_result_type",
          function(object)
	  {
	    return(object@posteriors)
	  })


#' @rdname bagel-accessors
#' @usage NULL
setMethod("taus","bagel_result_type",
          function(object)
	  {
	    return(object@taus)
	  })



#' @rdname bagel-accessors
#' @usage NULL
setMethod("ratios","bagel_result_type",
          function(object)
	  {
	    return(object@ratios)	
	  })


#' @rdname bagel-accessors
#' @usage NULL
setMethod("weights_tau_zero","bagel_result_type",
          function(object)
	  {
	    return(object@w0ts)
	  })



#' @rdname bagel-accessors
#' @usage NULL
setMethod("time","bagel_result_type",
          function(object)
	  {
	    return(object@t - 1)
	  })

#' @rdname bagel-accessors
#' @usage NULL
setMethod("max_particles","bagel_result_type",
          function(object)
	  {
	    return(object@max_particles)
	  })


#' MAP changepoint location and alarm time
#'
#' Once the change is occured, one might want to find the most probable change location. The location is the
#' maximum a posteriori (MAP) estimate conditional on a change.
#'
#' @param object A fitted result returned by [bagel_kv_exact()] or a related
#'   batch fitting function.
#' @returns A list with integer components `location` and `detected`.
#'   `location` is available even when no alarm occurred, including runs
#'   with `threshold = Inf`. It is `NA_integer_` if there are no
#'   positive-weight change candidates. Ties use the first maximum.
#'   `detected` is the first observation whose change probability strictly
#'   exceeds the threshold, or `NA_integer_` if none does. An alarm at
#'   the final observation is included.
#' @details
#' With `"bagel"` or `"LP-M"`, a change at `location = k` starts
#' at observation `k + 1`. The `"LP-P"` indexing exception is described below.
#' The alarm occurs when the posterior change probability exceeds the
#' threshold. A threshold of one or
#' greater, including `Inf`, disables early stopping. In the internal functions, the weights and 
#' thresholds are compared in the logarithmic scale for numerical stability, but the returned values are all based on the original scale.
#'
#' A location estimate can be returned even when there was no alarm. It uses
#' all observations processed so far. We identify a run without an alarm
#' using `is.na(changepoint(object)$detected)`; the function always returns
#' a list.
#'
#' With `"bagel"` or `"LP-M"`, [weights()] represents all candidate
#' locations. `"LP-P"` removes some candidates, so `location` then gives a
#' position among the remaining candidates rather than the original time.
#' If a positive-weight change candidate remains, we recover its original
#' boundary from `taus(object)[-1]` at the maximum of
#' `active_weights(object)[-1]`.
#' @seealso [bagel_kv_exact()], [weights()], [taus()]
#' @examples
#' feature <- function(t, tau) matrix(if (tau == 0) 1 else c(1, 1), ncol = 1)
#' prior <- function(t) {
#'   if (t == 1) list(mu = matrix(0), sigma = matrix(1))
#'   else list(mu = matrix(0, 2, 1), sigma = matrix(c(1, -1, -1, 2), 2))
#' }
#' transform <- function(tau) matrix(if (tau == 0) 1 else c(1, 1), nrow = 1)
#' fit <- bagel_kv_exact(
#'   c(0, 0.1, -0.1, 3, 3.1), feature, prior, transform,
#'   p0 = 0.9, p = 1, sigma = 1, max_particles = 5, threshold = 0.95)
#' changepoint(fit)
#' is.na(changepoint(fit)$detected)
setGeneric("changepoint",function(object) standardGeneric("changepoint"))
#' @rdname changepoint
#' @usage NULL
setMethod("changepoint","bagel_result_type",
          function(object)
          {
            change_weights <- weights(object)[-1]
            location <- if (length(change_weights) && any(change_weights > 0)) {
              which.max(change_weights)
            } else NA_integer_

            log_cutoff <- if (object@threshold >= 1) Inf else
              -log1p(-object@threshold)
            alarms <- which(-object@log_w0ts > log_cutoff)
            detected <- if (length(alarms)) alarms[1] else NA_integer_

            list(location = location, detected = detected)
          })

