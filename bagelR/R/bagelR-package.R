#' Bayesian online changepoint detection for linear models
#'
#' Bagel is a fast Bayesian online changepoint detection algorithm for linear models via posterior merging.
#' It can detect changes in time series in sequential mode or in batch mode. 
#' To use it, we need to specify the feature, prior, and transformation functions and set thresholds.
#' The vignette introduces these functions through mean, slope, seasonal, and
#' autocorrelated-error examples:
#' `vignette("bagelR", package = "bagelR")`.
#'
#' @section Choosing a model:
#' For a change in mean with known observation variance, we use
#' [bagel_kv_exact()]. For known-variance models with two or more coefficients
#' in the base (no-change) model, we use [bagel_kv_approximate()]. When the
#' observation variance is unknown, we use [bagel_uv_approximate()].
#' @section Reading the results:
#' For a fitted result `fit`, `1 - weights_tau_zero(fit)` gives the posterior
#' probability of a change at each processed observation. We obtain the
#' estimated change location and the first alarm time with `changepoint(fit)`.
#' Coefficient estimates for each retained group are available through
#' `active_posteriors(fit)`, which returns posterior means and covariance
#' factors. Under posterior merging, each group represents one or more
#' candidate change locations; the no-change particle is listed first.
#'
#' We use `weights(fit)` to inspect the recovered posterior distribution over all changepoint candidates. 
#' The [results page][bagel-accessors]
#' describes these quantities in detail.
#' @section Processing arriving observations:
#' We create a detector with [sequential_bagel_kv_exact()] or a related
#' sequential constructor and pass each observation to [update()]. Its
#' return value is the current no-change weight. We record `1 - no_change`
#' and apply a stopping rule in the observation loop, as in the vignette.
#'
#' We use `bagelR::update()`, `bagelR::time()`, and `bagelR::weights()` to
#' distinguish these functions from functions with the same names in other
#' packages.
#'
#' @references
#' Yang, Z., Fearnhead, P., and Grose, D. (2026).
#' Bagel: A Fast Bayesian Online Changepoint Detection Algorithm for Linear
#' Models via Posterior Merging. <https://arxiv.org/abs/2611.03230v1>.
#' @useDynLib bagelR, .registration = TRUE
#' @import methods Rcpp
#' @export bagel_kv_exact bagel_kv_approximate bagel_uv_approximate
#' @export sequential_bagel_kv_exact sequential_bagel_kv_approximate
#' @export sequential_bagel_uv_approximate update changepoint
#' @export weights active_weights active_posteriors taus ratios weights_tau_zero
#' @export time max_particles mus sigmas nu iota
#' @keywords package
"_PACKAGE"
