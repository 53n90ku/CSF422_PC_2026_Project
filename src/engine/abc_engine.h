#ifndef ABC_ENGINE_H
#define ABC_ENGINE_H

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "../rng.h"

struct RunResult {
    double runtime_s;
    double throughput;
    int n_total;
    int n_accepted;
    double acceptance_rate;
    std::vector<std::vector<double>> params;  // params[j][i]
    std::vector<int> steps;
};

template <typename Model>
RunResult abc_run(const Model& model, int n_particles, double epsilon,
                  uint64_t seed) {
    Rng rng = rng_seed(seed);
    auto s_obs = model.observed_stats();
    int n_params = static_cast<int>(Model::param_names().size());

    RunResult result;
    result.n_total = n_particles;
    result.params.resize(n_params);
    result.steps.reserve(n_particles);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_particles; ++i) {
        auto theta = model.sample_prior(rng);
        auto sim = model.simulate_full(theta, rng);
        double d = model.distance(sim.summary, s_obs);
        result.steps.push_back(sim.steps);

        if (d <= epsilon) {
            std::vector<double> row;
            row.reserve(n_params);
            Model::pack_params(theta, row);
            for (int c = 0; c < n_params; ++c) {
                result.params[c].push_back(row[c]);
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    result.runtime_s = elapsed;
    result.n_accepted = static_cast<int>(result.params[0].size());
    result.acceptance_rate = static_cast<double>(result.n_accepted) / n_particles;
    result.throughput = n_particles / elapsed;
    return result;
}

inline bool summary_file_has_header(const std::string& path) {
    std::ifstream in(path);
    return in.good() && in.peek() != std::ifstream::traits_type::eof();
}

inline void write_summary_csv(const std::string& path,
                              const std::vector<RunResult>& runs,
                              bool append = false) {
    const bool use_append = append || summary_file_has_header(path);
    std::ofstream f(path, use_append ? std::ios::app : std::ios::out);
    if (!use_append) {
        f << "n_particles,runtime_s,throughput_ps,n_accepted,acceptance_rate,"
             "steps_mean,steps_std\n";
    }
    for (auto& r : runs) {
        double s_mean = 0, s_std = 0;
        if (!r.steps.empty()) {
            s_mean = std::accumulate(r.steps.begin(), r.steps.end(), 0.0)
                     / r.steps.size();
            double sq_sum = 0;
            for (int s : r.steps) sq_sum += (s - s_mean) * (s - s_mean);
            s_std = std::sqrt(sq_sum / r.steps.size());
        }
        f << r.n_total << ","
          << std::fixed << std::setprecision(6) << r.runtime_s << ","
          << std::fixed << std::setprecision(1) << r.throughput << ","
          << r.n_accepted << ","
          << std::fixed << std::setprecision(6) << r.acceptance_rate << ","
          << std::fixed << std::setprecision(2) << s_mean << ","
          << std::fixed << std::setprecision(2) << s_std << "\n";
    }
}

template <typename Model>
void write_samples_csv(const std::string& path, const RunResult& result) {
    std::ofstream f(path);
    auto names = Model::param_names();
    for (size_t c = 0; c < names.size(); ++c) {
        if (c > 0) f << ",";
        f << names[c];
    }
    f << "\n";

    int n_accepted = result.n_accepted;
    int n_params = static_cast<int>(names.size());
    for (int i = 0; i < n_accepted; ++i) {
        for (int c = 0; c < n_params; ++c) {
            if (c > 0) f << ",";
            f << std::fixed << std::setprecision(8) << result.params[c][i];
        }
        f << "\n";
    }
}

inline double vec_mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

inline double vec_var(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double m = vec_mean(v);
    double sq = 0;
    for (double x : v) sq += (x - m) * (x - m);
    return sq / (v.size() - 1);
}

inline void print_posterior_summary(const RunResult& result,
                                    const std::vector<std::string>& names) {
    std::cout << "\nposterior:\n";
    for (size_t c = 0; c < names.size(); ++c) {
        std::cout << "  " << names[c]
                  << ": mean = " << std::fixed << std::setprecision(6)
                  << vec_mean(result.params[c])
                  << "  var = " << vec_var(result.params[c]) << "\n";
    }
}

#endif
