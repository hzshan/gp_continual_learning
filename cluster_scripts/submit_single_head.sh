script_name=single_head.py

P_test=500
n_tasks=30
T=0
sigma=0.2
depth=1
fixed_w=0
rotation=0
dist=0
teacher_speed=1
N0=1000
cycles=0


MEM_REQUEST=10000 # memory requested, in MB
TIME_REQUEST=0-10:30
PARTITION=shared

batch_name=${n_tasks}tasks_TS${teacher_speed}_R${rotation}_D${dist}_${depth}L_fW${fixed_w}_${cycles}cycles
#batch_name=cifar_debug2


directory="/n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/"
rm -rf $directory  # remove the directory
mkdir $directory

values=(500)
#values=(1)
seeds=(0 1 2 3 4 5 6 7 8 9)
#values="$(seq 0 0.1 1.57)"
trial_ind=0
# values=(400 800 1600 3200)


# shellcheck disable=SC2068

for P in "${values[@]}"
do
  echo $batch_name
  echo $P
  export P P_test n_tasks T sigma depth fixed_w \
  script_name batch_name trial_ind rotation teacher_speed dist N0 cycles

  for seed in "${seeds[@]}"
  do
    export seed
    sbatch --account=cox_lab --job-name=$batch_name --mem=$MEM_REQUEST -t $TIME_REQUEST -p $PARTITION\
    -o /n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/run_message_${trial_ind}.txt\
    /n/home11/haozheshan/ContinualLearning2022/cluster_scripts/single_head.sbatch.sh

    trial_ind=$((trial_ind+1))
    sleep 0.5
  done
done
