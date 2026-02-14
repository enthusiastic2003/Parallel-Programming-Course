#!/bin/bash
#SBATCH --job-name=cudalcs              # Job name
#SBATCH --partition=debug              # Partition name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=32             # Number of CPU cores per task
#SBATCH --output=outputcudalcs.log           # Output file
#SBATCH --error=errorcudalcs.log             # Error log file

# Explicitly set Lmod path
export PATH=/usr/local/lmod:$PATH
module load cuda

# Compile and run the CUDA application with 5 different seeds
cd /home/sirjanhansda/openmp_tests

nvcc ./cuda_prog.cu -o ./cuda_lcs

echo "Running with 5 different seeds..."
./cuda_lcs 123456  
./cuda_lcs 548621
./cuda_lcs 589123 
./cuda_lcs 268421 
./cuda_lcs 784152 