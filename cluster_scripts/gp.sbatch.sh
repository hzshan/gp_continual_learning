#!/bin/bash
#!/bin/bash
#SBATCH -c 1                # Number of cores (-c)

module load python
module load cuda/9.0-fasrc02
source activate continual
cd /n/home11/haozheshan/ContinualLearning2022 || exit
python /n/home11/haozheshan/ContinualLearning2022/$script_name  \
--TRIAL_IND $trial_ind \
--BATCH_NAME $batch_name \
--P $P \
--naive_gp $naive_gp \
--P_test $P_test \
--n_tasks $n_tasks \
--lambda_val $lambda_val \
--use_large_lambda_limit $use_large_lambda_limit \
--manipulation_ratio $manipulation_ratio \
--resample $resample \
--T $T \
--resample $resample \
--sigma $sigma \
--depth $depth \
--task_type $task_type \
--dataset $dataset \
--seed $seed \
--N0context $N0context \
--context_strength $context_strength \
--whiten $whiten \
--cluster 1 \
--save_outputs $save_outputs \
--only_first_task $only_first_task