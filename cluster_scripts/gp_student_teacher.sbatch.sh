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
--N0 $N0 \
--Nh $Nh \
--NC $NC \
--radius $radius \
--tsim $tsim \
--xsim $xsim \
--depth $depth \
--lambda_val $lambda_val \
--NSEEDS $NSEEDS \
--N0context $N0context \
--context_strength $context_strength \
--change_w_in_teachers $change_w_in_teachers \
--use_large_lambda_limit $use_large_lambda_limit \
--input_share_variability $input_share_variability \
--cluster 1