#ifndef SIR_MODEL_H
#define SIR_MODEL_H

// Discrete-time SIR. Params (beta, gamma), T=160, N=1000.
// early_exit: stop when I hits 0. Otherwise always run T steps.

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "../rng.h"

struct SIRParams {
    double beta;
    double gamma;
};

struct SIRSummary {
    double peak_infected;
    double time_of_peak;
    double final_recovered;
};

struct SIRCalibration {
    SIRSummary s_obs;
    double sigma[3];
    double epsilon;
};

struct SIRModel {
    using Params  = SIRParams;
    using Summary = SIRSummary;

    struct SimResult {
        Summary summary;
        int steps;
    };

    static constexpr int    N_POP = 1000;
    static constexpr int    S0    = 990;
    static constexpr int    I0    = 10;
    static constexpr int    R0    = 0;
    static constexpr int    T_MAX = 160;
    static constexpr int    N_PARAMS = 2;

    static constexpr double BETA_LO  = 0.1,  BETA_HI  = 1.0;
    static constexpr double GAMMA_LO = 0.01, GAMMA_HI = 0.5;

    bool early_exit = true;
    SIRCalibration calib = {};

    static std::vector<std::string> param_names() {
        return {"beta", "gamma"};
    }

    Params sample_prior(Rng& rng) const {
        return {
            rng_uniform(rng, BETA_LO, BETA_HI),
            rng_uniform(rng, GAMMA_LO, GAMMA_HI)
        };
    }

    SimResult simulate_full(const Params& p, Rng& rng) const {
        int S = S0, I = I0, R = R0;
        int peak_I = I;
        int peak_t = 0;
        int steps = 0;

        for (int t = 1; t <= T_MAX; ++t) {
            double p_inf = 1.0 - std::exp(-p.beta * I / static_cast<double>(N_POP));
            int new_inf = rng_binomial(rng, S, p_inf);
            int new_rec = rng_binomial(rng, I, p.gamma);

            S -= new_inf;
            I += new_inf - new_rec;
            R += new_rec;

            if (I > peak_I) {
                peak_I = I;
                peak_t = t;
            }
            steps = t;
            if (early_exit && I <= 0) break;
        }

        Summary s;
        s.peak_infected   = static_cast<double>(peak_I);
        s.time_of_peak    = static_cast<double>(peak_t);
        s.final_recovered = static_cast<double>(R);
        return {s, steps};
    }

    Summary observed_stats() const {
        return calib.s_obs;
    }

    double distance(const Summary& sim, const Summary& obs) const {
        double d0 = (sim.peak_infected   - obs.peak_infected)   / calib.sigma[0];
        double d1 = (sim.time_of_peak    - obs.time_of_peak)    / calib.sigma[1];
        double d2 = (sim.final_recovered - obs.final_recovered) / calib.sigma[2];
        return std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
    }

    static void pack_params(const Params& p, std::vector<double>& out) {
        out.push_back(p.beta);
        out.push_back(p.gamma);
    }

    static Summary generate_observed(double beta_true, double gamma_true,
                                     uint64_t seed) {
        SIRModel model;
        model.early_exit = true;
        Rng rng = rng_seed(seed);
        Params p_true = {beta_true, gamma_true};
        return model.simulate_full(p_true, rng).summary;
    }

    struct PilotResult {
        double sigma[3];
        double epsilon;
        std::vector<double> distances;
    };

    static PilotResult run_pilot(int n_pilot, const Summary& s_obs_ref,
                                 uint64_t seed, double quantile = 0.02) {
        SIRModel model;
        model.early_exit = true;
        Rng rng = rng_seed(seed);

        std::vector<double> peaks, times, recovereds;
        peaks.reserve(n_pilot);
        times.reserve(n_pilot);
        recovereds.reserve(n_pilot);

        for (int i = 0; i < n_pilot; ++i) {
            Params theta = model.sample_prior(rng);
            auto res = model.simulate_full(theta, rng);
            peaks.push_back(res.summary.peak_infected);
            times.push_back(res.summary.time_of_peak);
            recovereds.push_back(res.summary.final_recovered);
        }

        auto std_dev = [](const std::vector<double>& v) {
            double m = 0;
            for (double x : v) m += x;
            m /= v.size();
            double sq = 0;
            for (double x : v) sq += (x - m) * (x - m);
            return std::sqrt(sq / v.size());
        };

        PilotResult pr;
        pr.sigma[0] = std_dev(peaks);
        pr.sigma[1] = std_dev(times);
        pr.sigma[2] = std_dev(recovereds);
        for (int k = 0; k < 3; ++k) {
            if (pr.sigma[k] < 1e-12) pr.sigma[k] = 1.0;
        }

        Rng rng2 = rng_seed(seed);
        pr.distances.reserve(n_pilot);
        for (int i = 0; i < n_pilot; ++i) {
            Params theta = model.sample_prior(rng2);
            auto res = model.simulate_full(theta, rng2);
            double d0 = (res.summary.peak_infected   - s_obs_ref.peak_infected)   / pr.sigma[0];
            double d1 = (res.summary.time_of_peak    - s_obs_ref.time_of_peak)    / pr.sigma[1];
            double d2 = (res.summary.final_recovered - s_obs_ref.final_recovered) / pr.sigma[2];
            pr.distances.push_back(std::sqrt(d0 * d0 + d1 * d1 + d2 * d2));
        }

        std::vector<double> sorted = pr.distances;
        std::sort(sorted.begin(), sorted.end());
        int idx = static_cast<int>(quantile * sorted.size());
        if (idx >= static_cast<int>(sorted.size())) {
            idx = static_cast<int>(sorted.size()) - 1;
        }
        pr.epsilon = sorted[idx];
        return pr;
    }
};

#endif
