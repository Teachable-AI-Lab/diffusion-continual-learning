## Usage
- `unit-test.ipynb` is the main logic to 1. train DDPM with EWC and 2. analysis Fisher information matrix.
- You want to first train DDPM on task 0, before applying EWC and training on task 1
- We have three versions of Fisher used in EWC: diagnoal, rank-1 and rank-1 optimal. `compute_rank1_coeff_and_mean` function computes all necessary values.
  - `mu` is the rank-1 vector
- You want to compute the Fisher by inferencing on the old task
- The `test FID` block is the main evaluation metric. Say the model is sequentially trained on task 0 and task 1. We want to evaluate the model's generation quality on task 0 and 1. Here we compare against the test set.
- To emprically analysis the **full** Fisher information matrix, we need a small DDPM by uncommentting the configurations in the model initialization step.