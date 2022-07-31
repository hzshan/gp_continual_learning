script_name=main.py
sbatch_name=-
P=100
P_test=100
n_tasks=10
N0=1000
T=0
sigma=0.2
student_type='relu'
NUM_SEEDS=10
fixed_w=True
teacher_speed=0
input_rotation=0
input_dist=10


MEM_REQUEST=1000 # memory requested, in MB
TIME_REQUEST=0-0:30
PARTITION=serial_requeue

# OVERWRITE THE DEFAULT OPTIONS
#rand_id=$RANDOM

batch_name='reluST_diff_rotation_speed'

#values=(50)
values="$(seq 0 0.1 1.57)"
trial_ind=0
# values=(400 800 1600 3200)

# shellcheck disable=SC2068
for input_rotation in ${values[@]}
do
  echo $input_rotation
  export P P_test n_tasks N0 T sigma student_type NUM_SEEDS fixed_w teacher_speed\
 input_rotation input_dist script_name batch_name trial_ind
  sbatch --account=cox_lab --job-name=$batch_name --mem=$MEM_REQUEST -t $TIME_REQUEST -p $PARTITION\
 -o /n/home11/haozheshan/ContinualLearning2022/test.txt\
 /n/home11/haozheshan/ContinualLearning2022/sbatch.sh


  trial_ind=$((trial_ind+1))
  sleep 0.5
done



