#!/bin/bash
#!/bin/bash
#SBATCH -c 1                # Number of cores (-c)

module load python
module load cuda/9.0-fasrc02
source activate continual
cd /n/home11/haozheshan/ContinualLearning2022 || exit
python $script_name  \
--TRIAL_IND $trial_ind \
--BATCH_NAME $batch_name \
--P $P \
--P_test $P_test \
--n_tasks $n_tasks \
--N0 $N0 \
--T $T \
--sigma $sigma \
--student_type $student_type \
--NUM_SEEDS $NUM_SEEDS \
--fixed_w $fixed_w \
--teacher_speed $teacher_speed \
--input_rotation $input_rotation \
--input_dist $input_dist
