#!/bin/bash
#SBATCH --job-name=openmp32              # Job name
#SBATCH --partition=debug              # Partition name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=32             # Number of CPU cores per task
#SBATCH --output=output32_t8.log           # Output file
#SBATCH --error=error32.log             # Error log file

# Explicitly set Lmod path
export PATH=/usr/local/lmod:$PATH

# Compile and run the OpenMP application with 5 different seeds
cd /home/sirjanhansda/openmp_tests

echo "Running with 5 different seeds..."
echo "Total allocated SLURM CPUS: ($SLURM_CPUS_PER_TASK)"
export OMP_NUM_THREADS=8
./myapp 123456 
./myapp 548621 
./myapp 589123 
./myapp 268421 
./myapp 784152 