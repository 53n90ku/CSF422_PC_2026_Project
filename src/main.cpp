#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <sys/stat.h>

#include "rng.h"
#include "models/gaussian_model.h"
#include "models/sir_model.h"
#include "engine/abc_engine.h"

struct Args {
    std::string model = "gaussian";
    int n = 10000;
    double epsilon = 0.1;
    uint64_t seed = 42;
    bool early_exit = false;
    int reps = 1;
    std::string out_dir = "../results";
    int pilot_n = 5000;
    double pilot_q = 0.02;
};

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "  --model MODEL     gaussian, sir, or sir-pilot\n"
        << "  -N COUNT          particle count (default 10000)\n"
        << "  --epsilon EPS     tolerance (default 0.1; SIR uses calib if unset)\n"
        << "  --seed SEED       rng seed (default 42)\n"
        << "  --early-exit      stop SIR when I = 0\n"
        << "  --reps R          repeats (default 1)\n"
        << "  --out-dir DIR     csv folder (default ../results)\n"
        << "  --pilot-n N       SIR pilot size (default 5000)\n"
        << "  --pilot-q Q       SIR epsilon quantile (default 0.02)\n";
}

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) a.model = argv[++i];
        else if (arg == "-N" && i + 1 < argc) a.n = std::atoi(argv[++i]);
        else if (arg == "--epsilon" && i + 1 < argc) a.epsilon = std::atof(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) a.seed = std::strtoull(argv[++i], nullptr, 10);
        else if (arg == "--early-exit") a.early_exit = true;
        else if (arg == "--reps" && i + 1 < argc) a.reps = std::atoi(argv[++i]);
        else if (arg == "--out-dir" && i + 1 < argc) a.out_dir = argv[++i];
        else if (arg == "--pilot-n" && i + 1 < argc) a.pilot_n = std::atoi(argv[++i]);
        else if (arg == "--pilot-q" && i + 1 < argc) a.pilot_q = std::atof(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }
    return a;
}

static void ensure_dir(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

static void run_gaussian(const Args& a) {
    std::cout << "Gaussian ABC  N=" << a.n
              << "  eps=" << a.epsilon
              << "  seed=" << a.seed
              << "  reps=" << a.reps << "\n";

    ensure_dir(a.out_dir);
    std::vector<RunResult> runs;
    for (int r = 0; r < a.reps; ++r) {
        auto result = abc_run(GaussianModel{}, a.n, a.epsilon, a.seed + r);
        std::cout << "rep " << r + 1 << ": " << result.runtime_s << " s, "
                  << result.n_accepted << " accepted ("
                  << result.acceptance_rate * 100.0 << "%)\n";
        print_posterior_summary(result, GaussianModel::param_names());
        runs.push_back(std::move(result));
    }

    write_summary_csv(a.out_dir + "/gaussian_summary.csv", runs);
    write_samples_csv<GaussianModel>(
        a.out_dir + "/gaussian_samples_" + std::to_string(a.n) + ".csv",
        runs.back());
}

static void run_sir_pilot(const Args& a) {
    ensure_dir(a.out_dir);

    double beta_true = 0.4, gamma_true = 0.1;
    SIRSummary s_obs = SIRModel::generate_observed(beta_true, gamma_true, a.seed);

    std::cout << "SIR observed (beta=" << beta_true
              << ", gamma=" << gamma_true << "): peak=" << s_obs.peak_infected
              << " t=" << s_obs.time_of_peak
              << " R=" << s_obs.final_recovered << "\n";

    {
        std::ofstream f(a.out_dir + "/sir_observed.csv");
        f << "peak_infected,time_of_peak,final_recovered\n"
          << s_obs.peak_infected << ","
          << s_obs.time_of_peak << ","
          << s_obs.final_recovered << "\n";
    }

    auto pilot = SIRModel::run_pilot(a.pilot_n, s_obs, a.seed + 1000, a.pilot_q);
    std::cout << "pilot n=" << a.pilot_n
              << "  sigma=[" << pilot.sigma[0] << ", " << pilot.sigma[1]
              << ", " << pilot.sigma[2] << "]"
              << "  eps=" << pilot.epsilon << "\n";

    {
        std::ofstream f(a.out_dir + "/sir_calibration.csv");
        f << "s_obs_peak,s_obs_time,s_obs_recovered,"
             "sigma_peak,sigma_time,sigma_recovered,epsilon\n"
          << s_obs.peak_infected << ","
          << s_obs.time_of_peak << ","
          << s_obs.final_recovered << ","
          << pilot.sigma[0] << "," << pilot.sigma[1] << ","
          << pilot.sigma[2] << "," << pilot.epsilon << "\n";
    }
}

static bool load_sir_calibration(const std::string& path, SIRCalibration& calib) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string header;
    std::getline(f, header);
    char comma;
    f >> calib.s_obs.peak_infected >> comma
      >> calib.s_obs.time_of_peak >> comma
      >> calib.s_obs.final_recovered >> comma
      >> calib.sigma[0] >> comma
      >> calib.sigma[1] >> comma
      >> calib.sigma[2] >> comma
      >> calib.epsilon;
    return f.good() || f.eof();
}

static void run_sir(const Args& a) {
    SIRModel model;
    model.early_exit = a.early_exit;

    std::string calib_path = a.out_dir + "/sir_calibration.csv";
    if (!load_sir_calibration(calib_path, model.calib)) {
        std::cerr << "missing " << calib_path << " (run --model sir-pilot first)\n";
        std::exit(1);
    }

    double eps = (a.epsilon != 0.1) ? a.epsilon : model.calib.epsilon;
    std::cout << "SIR ABC  mode=" << (a.early_exit ? "early-exit" : "fixed-T")
              << "  N=" << a.n << "  eps=" << eps << "  seed=" << a.seed
              << "  reps=" << a.reps << "\n";

    ensure_dir(a.out_dir);
    std::vector<RunResult> runs;
    for (int r = 0; r < a.reps; ++r) {
        auto result = abc_run(model, a.n, eps, a.seed + r);

        double s_mean = 0, s_std = 0;
        if (!result.steps.empty()) {
            s_mean = std::accumulate(result.steps.begin(), result.steps.end(), 0.0)
                     / result.steps.size();
            double sq = 0;
            for (int s : result.steps) sq += (s - s_mean) * (s - s_mean);
            s_std = std::sqrt(sq / result.steps.size());
        }

        std::cout << "rep " << r + 1 << ": " << result.runtime_s << " s, "
                  << result.n_accepted << " accepted ("
                  << result.acceptance_rate * 100.0 << "%)"
                  << "  steps mean=" << s_mean << " std=" << s_std << "\n";
        print_posterior_summary(result, SIRModel::param_names());
        runs.push_back(std::move(result));
    }

    std::string tag = a.early_exit ? "sir_early" : "sir_fixed";
    write_summary_csv(a.out_dir + "/" + tag + "_summary.csv", runs);
    write_samples_csv<SIRModel>(
        a.out_dir + "/" + tag + "_samples_" + std::to_string(a.n) + ".csv",
        runs.back());
}

int main(int argc, char* argv[]) {
    Args a = parse_args(argc, argv);
    if (a.model == "gaussian") run_gaussian(a);
    else if (a.model == "sir-pilot") run_sir_pilot(a);
    else if (a.model == "sir") run_sir(a);
    else {
        std::cerr << "Unknown model: " << a.model << "\n";
        return 1;
    }
    return 0;
}
