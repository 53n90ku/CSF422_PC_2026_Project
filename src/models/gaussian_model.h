#ifndef GAUSSIAN_MODEL_H
#define GAUSSIAN_MODEL_H

// theta ~ N(0,1), y ~ N(theta,1), y_obs = 1, d = |y - 1|
// exact posterior at eps=0 is N(0.5, 0.5)

#include <cmath>
#include <string>
#include <vector>

#include "../rng.h"

struct GaussianParams {
    double theta;
};

struct GaussianSummary {
    double y;
};

struct GaussianModel {
    using Params  = GaussianParams;
    using Summary = GaussianSummary;

    struct SimResult {
        Summary summary;
        int steps;
    };

    static constexpr double Y_OBS    = 1.0;
    static constexpr int    N_PARAMS = 1;

    static std::vector<std::string> param_names() {
        return {"theta"};
    }

    Params sample_prior(Rng& rng) const {
        return {rng_normal(rng, 0.0, 1.0)};
    }

    Summary simulate(const Params& p, Rng& rng) const {
        return {rng_normal(rng, p.theta, 1.0)};
    }

    Summary observed_stats() const {
        return {Y_OBS};
    }

    double distance(const Summary& sim, const Summary& obs) const {
        return std::fabs(sim.y - obs.y);
    }

    SimResult simulate_full(const Params& p, Rng& rng) const {
        return {simulate(p, rng), 1};
    }

    static void pack_params(const Params& p, std::vector<double>& out) {
        out.push_back(p.theta);
    }
};

#endif
