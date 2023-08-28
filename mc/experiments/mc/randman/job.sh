#! /bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=10:00:00

# set name of job

# set number of GPUs
#SBATCH --gres=gpu:1

# set the partition to use
#SBATCH --partition=small

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address

# run the application
module load python/anaconda3 gcc/9.1.0 use.dev cuda/11.0 swig openmpi/4.0.5-gcc-9.1.0 nano
source activate test

cd ../../../
python3 -m experiments.randman.run ${SLURM_ARRAY_TASK_ID} $1
