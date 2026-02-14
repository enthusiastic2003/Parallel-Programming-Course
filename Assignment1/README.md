
# OpenMP and CUDA LCS

This directory contains:

- An OpenMP C++ implementation in `prog.cpp`
- A CUDA implementation in `cuda_prog.cu`
- Slurm scripts to build and run on the cluster

## Build locally

Use the provided Makefile:

```
make            # builds both myapp and cuda_lcs
make openmp     # builds myapp only
make cuda       # builds cuda_lcs only
make clean
```

## Run locally

Both programs accept an optional seed (default is `123456`).

Note: to keep the final program short, some logging steps were removed. On compiling and running the programs, only the final results and time taken are printed.

Note: we encountered an issue where a CUDA program compiled on the master/login node did not run on the compute nodes due to driver/toolchain mismatch. The Slurm scripts compile on the compute node to avoid this.

```
./myapp 123456
./cuda_lcs 123456
```

## Run on the cluster (Slurm)

Use the scripts in this directory:

- `slurmjob32.sh` runs OpenMP with a fixed `OMP_NUM_THREADS` value.
- `slurmjobCuda.sh` compiles and runs the CUDA program on the compute node.
- `slurmjobCudaProfile.sh` compiles and runs the CUDA program under Nsight Systems.

Submit with:

```
sbatch slurmjob32.sh
sbatch slurmjobCuda.sh
sbatch slurmjobCudaProfile.sh
```

Run Logs are present to the `logs/` directory. See `logs/README.md` for naming details.
