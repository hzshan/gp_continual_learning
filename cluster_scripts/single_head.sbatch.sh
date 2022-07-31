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
--N0 $N0 \
--n_tasks $n_tasks \
--T $T \
--sigma $sigma \
--depth $depth \
--seed $seed \
--fixed_w $fixed_w \
--rotation $rotation \
--dist $dist \
--teacher_speed $teacher_speed \
--cycles $cycles \
--cluster 1