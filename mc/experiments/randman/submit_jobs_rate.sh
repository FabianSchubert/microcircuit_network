#!/usr/bin/bash

NUM_JOBS=$1

for((i=0; i<NUM_JOBS; i++))
do
	sbatch -J "sweep_$i" job_rate.sh $i $NUM_JOBS
done
