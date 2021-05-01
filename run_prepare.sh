#!/bin/bash
#SBATCH --job-name=TestGPUOnSaga
#SBATCH --account=nn9866k
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=512M
#SBATCH --partition=accel
#SBATCH --gres=gpu:1

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module load PyTorch/1.4.0-fosscuda-2019b-Python-3.7.4

# We got all the dependencies loaded for PyTorch but we don't want this version
# so unload it.
module unload PyTorch/1.4.0-fosscuda-2019b-Python-3.7.4

module list
source $SLURM_SUBMIT_DIR/env/bin/activate
# Setup monitoring
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory \
	--format=csv --loop=1 > "gpu_util-$SLURM_JOB_ID.csv" &
NVIDIA_MONITOR_PID=$!  # Capture PID of monitoring process
# Run our computation
python train_tacotron.py --force_align
# After computation stop monitoring
kill -SIGINT "$NVIDIA_MONITOR_PID"
