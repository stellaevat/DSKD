#!/bin/bash
#!
#! Example SLURM job script for Wilkes3 (AMD EPYC 7763, ConnectX-6, A100)
#! Last updated: Fri 30 Jul 11:07:58 BST 2021
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J gpujob
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH -A MLMI-set56-SL2-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#! Note probably this should not exceed the total number of GPUs in use.
#SBATCH --ntasks=1
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#! Note that the job submission script will enforce no more than 32 cpus per GPU.
#SBATCH --gres=gpu:4
#! How much wallclock time will be required?
#SBATCH --time=01:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=NONE
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! Do not change:
#SBATCH -p ampere

#! sbatch directives end here (put any additional directives above this line)

#! ######################################################################################
#! ######################################################################################

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

source /home/${USER}/.bashrc
source /home/${USER}/miniforge3/bin/activate
mamba activate "/home/${USER}/miniforge3/envs/dskd"


GPUS=(0 1 2 3)
WORK_DIR=.
MASTER_PORT=66$(($RANDOM%90+10))
DEVICE=$(IFS=,; echo "${GPUS[*]}")

export CUDA_LAUNCH_BLOCKING=1

CKPT_PATH=${1}
BATCH_SIZE=${2-32}

for seed in 10 20 30 40 50
do
    bash ${WORK_DIR}/scripts/eval/eval_main.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${CKPT_PATH} dolly ${BATCH_SIZE} $seed
done
for seed in 10 20 30 40 50
do
    bash ${WORK_DIR}/scripts/eval/eval_main.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${CKPT_PATH} self-inst ${BATCH_SIZE} $seed
done
for seed in 10 20 30 40 50
do
    bash ${WORK_DIR}/scripts/eval/eval_main.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${CKPT_PATH} vicuna ${BATCH_SIZE} $seed
done
for seed in 10 20 30 40 50
do
    bash ${WORK_DIR}/scripts/eval/eval_main.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${CKPT_PATH} sinst/11_ ${BATCH_SIZE} $seed
done
for seed in 10 20 30 40 50
do
    bash ${WORK_DIR}/scripts/eval/eval_main.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${CKPT_PATH} uinst/11_ ${BATCH_SIZE} $seed 10000
done
