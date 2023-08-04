#!/usr/bin/bash

NUM_JOBS=$1

sbatch --array=0-$(($NUM_JOBS - 1)) job.sh $MODEL $NUM_JOBS

