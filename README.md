# Quirk plane search

From-scratch reproduction of the offline search in Knapen, Lou, Papucci, and Setford, *Tracking down Quirks at the Large Hadron Collider*, [arXiv:1708.02243](https://arxiv.org/abs/1708.02243). The PDF is in `../paper/1708.02243.pdf`.

The search looks for a quirk/anti-quirk pair bound by an unbreakable string. In the straight-string regime the two trajectories stay on one plane. The code integrates that equation of motion through an eight-layer ATLAS barrel and runs the paper's plane finder.

## Layout

```text
configs/     detector, cuts, and the Table I benchmarks
src/quirk_search/
  dynamics.py       Eq. (3) and the Eq. (4) solution
  detector.py       pixel + SCT barrel, merging, 45 mm beam spot
  kinematics.py     pair production and the 200 GeV trigger
  backgrounds.py    pileup hits and photon conversions
  plane_finder.py   seeding on layers 8 and 7, then the inward walk
  selection.py      signal region and the efficiency factorisation
  sensitivity.py    cross sections and the 3- and 5-event labels
scripts/       one benchmark point, or a mass-tension scan
tests/         equation of motion, plane finder, selection
results/       json and csv written by the scripts
```



## Run the tests

```bash
cd october_work/quirk_search
python -m unittest discover -s tests -v
```

Needs NumPy. No MadGraph or Pythia.

## Run a benchmark point

```bash
python scripts/run_benchmark.py --name dy_800 --lambda-ev 4000 --events 8
python scripts/scan_sensitivity.py
```

`run_benchmark.py` writes `results/benchmarks/`. The scan writes `results/figures/sensitivity.csv`.

## What is and is not the paper's Monte Carlo

Implemented from the paper:

- straight-string force, including the Lorentz force in a 2 T solenoid
- pixel centres in r-phi and z, and SCT r-phi smearing at twice the 17 um intrinsic accuracy. The millimetre-scale SCT stereo z uncertainty is not applied as a hit shift, because it is larger than the 0.1 mm thickness cut
- hit merging, a known primary vertex, and the 25 ns window
- the uncentred tensor, \Delta\phi < 0.1, \Delta z < 2\mathrm{cm}, the factor-of-three growth test, and the signal region \Delta s < 0.01\mathrm{cm}, \Delta w < 1\mathrm{cm}
- efficiency split into trigger, fiducial, and reconstruction factors
- 5 events as discovery and 3 events as the background-free exclusion, at 300\mathrm{fb}^{-1}

Approximations, because the paper's event samples were not released:

- kinematics are sampled here, not generated with MadGraph5 and Pythia8
- Drell-Yan is leading-order photon exchange with a toy parton density
- QCD is leading-order gg and q\bar q, scaled so a 173 GeV quark pair has the 13 TeV t\bar t cross section of 831 pb
- the 30% charged quirk-hadron fraction for coloured quirks is applied as a weight
- pileup is uniform unassociated hits, 20 per interaction per layer at \langle\mu\rangle = 50
- there is no detector material, and the HSCP and monojet curves on Figure 6 are not included

A run of this code will not reproduce Table I or Figure 6 digit for digit. It is the same search, with those inputs replaced.