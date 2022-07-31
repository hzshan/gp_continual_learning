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
--P_test $P_test \
--n_tasks $n_tasks \
--T $T \
--sigma $sigma \
--depth $depth \
--seed $seed \
--fixed_w $fixed_w \
--lambda_val $lambda_val \
--dataset $dataset \
--permutation $permutation \
--n_epochs $n_epochs \
--resample $resample \
--interpolate $interpolate \
--cluster 1
