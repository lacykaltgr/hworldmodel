#!/bin/bash

#SBATCH --job-name=a2c_atari
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/a2c_atari_%j.txt
#SBATCH --error=slurm_errors/a2c_atari_%j.txt

current_commit=$(git rev-parse --short HEAD)
project_name="torchrl-example-check-$current_commit"
group_name="dreamer_cheetah"

#export PYTHONPATH=$(dirname $(dirname $PWD))
xvfb-run -a python functional/train.py

# Capture the exit status of the Python command
exit_status=$?
# Write the exit status to a file
if [ $exit_status -eq 0 ]; then
  echo "${group_name}_${SLURM_JOB_ID}=success" >>> report.log
else
  echo "${group_name}_${SLURM_JOB_ID}=error" >>> report.log
fi
