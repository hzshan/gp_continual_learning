script_name=gp.py
dataset='mnist'
task_type='permuted'
P=500
P_test=500 #default 500
n_tasks=30
T=0
sigma=0.2
depth=3
resample=0
naive_gp=0
lambda_val=100000


MEM_REQUEST=1000 # memory requested, in MB
TIME_REQUEST=0-3:10
PARTITION=shared
batch_name=gp_${n_tasks}x${P}_${dataset}_${task_type}_diff_similarity
#batch_name=cifar_debug2


directory="/n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/"
rm -rf $directory  # remove the directory
mkdir $directory

values=$(seq 0 0.1 1)
#values=(1 2 3 4 5 6 7 8 9 10)
seeds=$(seq 1 1 50)
#values="$(seq 0 0.1 1.57)"
trial_ind=0
# values=(400 800 1600 3200)


# shellcheck disable=SC2068

for permutation in ${values[@]}
do
  echo $batch_name
  echo $permutation
  export P P_test n_tasks T sigma depth permutation\
  script_name batch_name trial_ind lambda_val task_type dataset resample naive_gp

  for seed in $seeds
  do
    export seed
    sbatch --account=cox_lab --job-name=$batch_name --mem=$MEM_REQUEST -t $TIME_REQUEST -p $PARTITION\
    -o /n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/run_message_${trial_ind}.txt\
    /n/home11/haozheshan/ContinualLearning2022/cluster_scripts/gp.sbatch.sh

    trial_ind=$((trial_ind+1))
    sleep 0.5
  done
done
