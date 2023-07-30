#!/bin/bash
#SBATCH -n 1                # Number of CPU cores (-c)
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --gpu-freq=high

module load python
module load cuda/9.0-fasrc02
source activate continual
cd /n/home11/haozheshan/ContinualLearning2022 || exit
python /n/home11/haozheshan/ContinualLearning2022/$script_name  \
--TRIAL_IND $trial_ind \
--BATCH_NAME $batch_name \
--P $P \
--P_test $P_test \
--resample $resample \
--N $N \
--n_tasks $n_tasks \
--eta $eta \
--decay $decay \
--minibatch $minibatch \
--T $T \
--sigma $sigma \
--depth $depth \
--seed $seed \
--l2 $l2 \
--dataset $dataset \
--task_type $task_type \
--permutation $permutation \
--n_steps $n_steps \
--cluster 1