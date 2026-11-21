# Rumination as self-supervised learning in a world model

Reproduce Figures 2–4 of *Computational model for Rumination as Self-Supervised
Learning in a World Model* from the model parameters and random seeds.
Figure 1 is a conceptual illustration and is outside the scope of this code.

## Reproduce the figures

Install Python 3.11.13, then clone or download this repository. Open a terminal
in the directory containing this README and run both commands.

```bash
python -m pip install -r requirements.txt
python scripts/reproduce_figures.py --output-dir figures
```

Use a new output directory for each run. The command writes three PDFs:

| File | Calculation |
| --- | --- |
| `fig2_positive_observations.pdf` | Trajectories and final distributions of perceived threat for three positive rumination gains. |
| `fig3_negative_observations.pdf` | Corresponding trajectories and distributions for three negative rumination gains. |
| `fig4_signed_interpretation.pdf` | Gain-dependent means and sample standard deviations, theoretical drifts, and simulated versus predicted distributions. |

Figures 2 and 3 use 1,024 trials per gain. They display the first 20 trajectories
and highlight the first three. Figure 4a uses 20 trials at each of 7,171 gain
pairs. Figures 4c and 4e use 16,384 trials per gain and compare the simulated
readout with the stationary Beta distribution after ten sessions (positive
gains) or with the finite-time Fokker–Planck solution after the first session
(negative gains). Each session comprises 10 defeat updates and 960
sensory-contact updates.

The representative ensembles in Figures 2 and 3 use the same ordered seeds
for all six gains. The parameter survey in Figure 4a uses a separate set of
20 seeds shared across all grid points. Their settings are recorded in
`configs/paper_ensemble.json`.

Figures 4c and 4e share a third set of random draws across all six gains.
The 16 blocks of 1,024 replicates use PCG64 seeds `990127 + 113 * k` for
`k = 0, …, 15`. For each replicate, the same normal and uniform variates drive
sensory contact at each common update. Positive-gain trajectories run for ten
sessions. Negative-gain trajectories stop after the first session. Both
panels use the same discrete-model update and initial state.
