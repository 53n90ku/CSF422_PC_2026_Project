# CS F422 Project — Milestone 1
# Sequential ABC (Gaussian + SIR)

Single-threaded C++ ABC rejection sampler. Later milestones add CUDA.

## Layout

- `src/` — sampler (`make` → `./abc`)
- `results/` — csv from the runs in the report
- `figures/` — plots used in the pdf
- `report/` — tex + `main.pdf`

## Build / run

```
cd src
make
./abc --model gaussian -N 10000 --epsilon 0.1 --seed 42 --reps 3
./abc --model sir-pilot --seed 42
./abc --model sir -N 10000 --seed 42 --early-exit
./abc --model sir -N 10000 --seed 42
```

SIR needs `results/sir_calibration.csv` from `sir-pilot` before sampling.
For Plots run : `python3 make_figures.py` from `src/`.
