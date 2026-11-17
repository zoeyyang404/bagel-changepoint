#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <map>
#include <memory>
#include <vector>
#include "bagel.h"

namespace py = pybind11;

namespace {
matrix as_matrix(py::handle value, const char* label, bool column = false) {
  auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
  if (!array || (array.ndim() != 2 && !(column && array.ndim() == 1)))
    throw py::value_error(std::string(label) + " must be a numeric matrix" +
                          (column ? " or a one-dimensional vector" : ""));
  const auto rows = array.shape(0);
  const auto cols = array.ndim() == 1 ? 1 : array.shape(1);
  if (rows == 0 || cols == 0 || (column && cols != 1))
    throw py::value_error(std::string(label) + " has an invalid shape");
  matrix result(rows, cols);
  for (py::ssize_t i = 0; i < rows; ++i)
    for (py::ssize_t j = 0; j < cols; ++j)
      result(i, j) = array.data()[i * cols + j];
  if (!result.allFinite())
    throw py::value_error(std::string(label) + " must contain only finite values");
  return result;
}

void positive(double value, const char* label) {
  if (!std::isfinite(value) || value <= 0)
    throw py::value_error(std::string(label) + " must be finite and positive");
}

struct detector_base {
  virtual ~detector_base() = default;
  virtual double update(double y) = 0;
  virtual int time() const = 0;
  virtual int max_particles() const = 0;
  virtual double log_w0() const = 0;
  virtual std::vector<double> active_weights() const = 0;
  virtual std::vector<double> log_weights() const = 0;
  virtual std::vector<int> taus() const = 0;
  virtual std::vector<std::vector<double>> ratios() const = 0;
  virtual py::list posteriors() const = 0;
  virtual void set_log_weights(const std::vector<double>& values) = 0;
  virtual void set_weights(const std::vector<double>& values) = 0;
  std::vector<double> weights() const {
    const auto active = active_weights();
    const auto relative = ratios();
    std::vector<double> recovered;
    for (size_t i = 0; i < active.size(); ++i)
      for (double fraction : relative[i]) recovered.push_back(active[i] * fraction);
    return recovered;
  }
};

template<typename Noise, KL_divergence_type Distance>
class detector final : public detector_base {
  using core_type = bagel_type<Noise, Distance>;
  py::function feature_, prior_, transform_;
  std::map<int, matrix> features_, transforms_;
  std::map<int, prior_type> priors_;
  std::unique_ptr<core_type> core_;
  bool updating_ = false;
  bool failed_ = false;

  void ready() const {
    if (failed_) throw std::runtime_error("An update failed inside the core; create a new detector.");
  }
  void populated() const {
    ready();
    if (core_->particles.empty()) throw py::value_error("Process an observation first.");
  }
  void mutable_state() const {
    populated();
    if (updating_) throw py::value_error("Cannot modify a detector from inside its model callbacks.");
  }

public:
  detector(py::function feature, py::function prior, py::function transform,
           double p0, double p, double sigma, double nu, double iota,
           int limit, const std::string& pruning)
    : feature_(std::move(feature)), prior_(std::move(prior)), transform_(std::move(transform)) {
    if (!std::isfinite(p0) || p0 <= 0 || p0 > 1)
      throw py::value_error("p0 must be in (0, 1]");
    positive(p, "p");
    if (limit < 3) throw py::value_error("max_particles must be at least 3");
    Noise noise;
    if constexpr (std::same_as<Noise, known_variance>) {
      positive(sigma, "sigma"); noise.sigma = sigma;
    } else {
      positive(nu, "nu"); positive(iota, "iota"); noise.nu = nu; noise.iota = iota;
    }
    model_type model;
    model.feature_vector_function = [this](const int&, const int& tau) { return features_.at(tau); };
    model.prior_function = [this](const int& t) { return priors_.at(t); };
    model.transformer_function = [this](const int& tau) { return transforms_.at(tau); };
    core_ = std::make_unique<core_type>(model, noise, p0, p, limit, parse_pruning_type(pruning));
  }
  detector(const detector&) = delete;
  detector& operator=(const detector&) = delete;

  double update(double y) override {
    ready();
    if (updating_) throw py::value_error("Cannot recursively update the same detector.");
    if (!std::isfinite(y)) throw py::value_error("y must be finite");
    struct update_guard {
      bool& flag;
      explicit update_guard(bool& f) : flag(f) { flag = true; }
      ~update_guard() { flag = false; }
    } guard(updating_);

    // Match the R adapter's callback indices and evaluation order. All
    // validation happens before mutating the numerical state.
    const int t = core_->t;
    std::vector<int> locations = taus();
    locations.push_back(t - 1);
    std::map<int, matrix> features, transforms;
    std::map<int, prior_type> priors;
    for (int tau : locations)
      features[tau] = as_matrix(feature_(t, tau), "feature_vector result", true);
    std::vector<int> times{1};
    times.insert(times.end(), locations.begin() + 1, locations.end());
    times.push_back(t);
    for (int index : times) {
      py::object value = prior_(index);
      if (!py::isinstance<py::dict>(value))
        throw py::value_error("prior must return a dict containing mu and sigma");
      const auto item = value.cast<py::dict>();
      if (!item.contains("mu") || !item.contains("sigma"))
        throw py::value_error("prior must return a dict containing mu and sigma");
      prior_type result;
      result.mu = as_matrix(item["mu"], "prior mu", true);
      result.sigma = as_matrix(item["sigma"], "prior sigma");
      if (result.sigma.rows() != result.mu.rows() || result.sigma.cols() != result.mu.rows() ||
          !result.sigma.isApprox(result.sigma.transpose(), 1e-10) ||
          result.sigma.llt().info() != Eigen::Success)
        throw py::value_error("prior sigma must be symmetric positive definite and match mu");
      priors[index] = result;
    }
    for (int tau : locations)
      transforms[tau] = as_matrix(transform_(tau), "transform result");
    const auto d = priors.at(1).mu.rows();
    const auto joint = priors.at(t).mu.rows();
    if (t > 1 && joint <= d)
      throw py::value_error("change priors must include additional change coefficients");
    for (const auto& particle : core_->particles) {
      const auto expected = particle.tau == 0 ? d : joint;
      if (particle.post.mu.rows() != expected)
        throw py::value_error("coefficient dimensions must remain constant across updates");
    }
    Eigen::Index transformed_dimension = -1;
    for (int tau : locations) {
      const auto expected = tau == 0 ? d : joint;
      if (features.at(tau).rows() != expected || transforms.at(tau).cols() != expected)
        throw py::value_error("feature and transform dimensions must match the coefficient prior");
      if (tau > 0) {
        if (transformed_dimension != -1 && transforms.at(tau).rows() != transformed_dimension)
          throw py::value_error("transforms must have the same output dimension for all change candidates");
        transformed_dimension = transforms.at(tau).rows();
      }
    }
    features_ = std::move(features);
    priors_ = std::move(priors);
    transforms_ = std::move(transforms);
    try { core_->update(y); }
    catch (...) { failed_ = true; throw; }
    return core_->weight_0_t();
  }
  int time() const override { ready(); return core_->t - 1; }
  int max_particles() const override { return core_->max_num_particles; }
  double log_w0() const override { populated(); return core_->log_weight_0_t(); }
  std::vector<double> active_weights() const override {
    ready(); std::vector<double> result;
    for (const auto& p : core_->particles) result.push_back(std::exp(p.log_weight));
    return result;
  }
  std::vector<double> log_weights() const override {
    ready(); std::vector<double> result;
    for (const auto& p : core_->particles) result.push_back(p.log_weight);
    return result;
  }
  std::vector<int> taus() const override {
    ready(); std::vector<int> result;
    for (const auto& p : core_->particles) result.push_back(p.tau);
    return result;
  }
  std::vector<std::vector<double>> ratios() const override {
    ready(); std::vector<std::vector<double>> result;
    for (const auto& p : core_->particles) result.emplace_back(p.ratios.begin(), p.ratios.end());
    return result;
  }
  py::list posteriors() const override {
    ready(); py::list result;
    for (const auto& p : core_->particles) {
      py::dict item;
      item["tau"] = p.tau; item["weight"] = std::exp(p.log_weight);
      // Return owned copies; arrays must outlive subsequent updates/detectors.
      item["mu"] = py::cast(matrix(p.post.mu));
      item["sigma"] = py::cast(matrix(p.post.sigma));
      if constexpr (std::same_as<Noise, unknown_variance>) {
        item["nu"] = p.noise_structure.nu; item["iota"] = p.noise_structure.iota;
      }
      result.append(item);
    }
    return result;
  }
  void set_log_weights(const std::vector<double>& values) override {
    mutable_state();
    if (values.size() != core_->particles.size())
      throw py::value_error("log_weights must have one value per active particle");
    for (double value : values)
      if (std::isnan(value) || value == std::numeric_limits<double>::infinity())
        throw py::value_error("log_weights must be finite or -Inf");
    auto it = core_->particles.begin();
    for (double value : values) { it->log_weight = value; ++it; }
  }
  void set_weights(const std::vector<double>& values) override {
    mutable_state();
    double sum = 0;
    std::vector<double> logs;
    for (double value : values) {
      if (!std::isfinite(value) || value < 0)
        throw py::value_error("weights must be finite and nonnegative");
      sum += value; logs.push_back(std::log(value));
    }
    if (std::abs(sum - 1) > 1e-10) throw py::value_error("weights must sum to one");
    set_log_weights(logs);
  }
};

std::unique_ptr<detector_base> make_detector(
    py::function feature, py::function prior, py::function transform,
    double p0, double p, int limit, const std::string& variance,
    const std::string& distance, double sigma, double nu, double iota,
    const std::string& pruning) {
  if (distance != "exact" && distance != "approximate")
    throw py::value_error("distance must be 'exact' or 'approximate'");
  if (variance == "known") {
    if (distance == "exact")
      return std::make_unique<detector<known_variance, exact>>(feature, prior, transform, p0, p, sigma, nu, iota, limit, pruning);
    return std::make_unique<detector<known_variance, approximate>>(feature, prior, transform, p0, p, sigma, nu, iota, limit, pruning);
  }
  if (variance == "unknown") {
    if (distance == "exact")
      return std::make_unique<detector<unknown_variance, exact>>(feature, prior, transform, p0, p, sigma, nu, iota, limit, pruning);
    return std::make_unique<detector<unknown_variance, approximate>>(feature, prior, transform, p0, p, sigma, nu, iota, limit, pruning);
  }
  throw py::value_error("variance must be 'known' or 'unknown'");
}
} // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "Native Bagel engine compiled from the shared C++ core.";
  py::class_<detector_base>(m, "Detector")
    .def(py::init(&make_detector), py::arg("feature_vector"), py::arg("prior"), py::arg("transform"),
         py::kw_only(), py::arg("p0") = 0.9, py::arg("p") = 1.0,
         py::arg("max_particles") = 100, py::arg("variance") = "known",
         py::arg("distance") = "exact", py::arg("sigma") = 1.0,
         py::arg("nu") = 3.0, py::arg("iota") = 2.0, py::arg("pruning") = "bagel")
    .def("update", &detector_base::update, py::arg("y"),
         "Process one observation in place and return the no-change weight.")
    .def_property_readonly("time", &detector_base::time)
    .def_property_readonly("max_particles", &detector_base::max_particles)
    .def_property_readonly("log_w0", &detector_base::log_w0)
    .def_property_readonly("no_change_probability", [](const detector_base& d) { return std::exp(d.log_w0()); })
    .def_property_readonly("active_weights", &detector_base::active_weights)
    .def_property_readonly("log_weights", &detector_base::log_weights)
    .def_property_readonly("weights", &detector_base::weights)
    .def_property_readonly("taus", &detector_base::taus)
    .def_property_readonly("ratios", &detector_base::ratios)
    .def_property_readonly("active_posteriors", &detector_base::posteriors)
    .def("set_weights", &detector_base::set_weights, py::arg("weights"))
    .def("set_log_weights", &detector_base::set_log_weights, py::arg("log_weights"));
}
