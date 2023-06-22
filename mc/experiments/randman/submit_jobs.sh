#!/usr/bin/bash

MODEL=$1

NUM_JOBS=$2

sbatch --array=0-$(($NUM_JOBS - 1)) job.sh $MODEL $NUM_JOBS

