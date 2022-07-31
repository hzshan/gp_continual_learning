script_name=batch_only.py
P=1000
P_test=500
n_tasks=10
T=0
sigma=0.2
permutation=0
dataset='fashion'
fixed_w=0
n_epochs=1
lambda_val=0.1


MEM_REQUEST=16000 # memory requested, in MB
TIME_REQUEST=0-4:30
PARTITION=shared

batch_name=${n_tasks}x${P}_${dataset}_permute${permutation}_BATCH_RESULTS
#batch_name=cifar_debug2


directory="/n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/"
rm -rf $directory  # remove the directory
mkdir $directory

values=(1 3 6 10)
#values=(1)
seeds=(0 1 2 3 4)
#values="$(seq 0 0.1 1.57)"
trial_ind=0
# values=(400 800 1600 3200)


# shellcheck disable=SC2068

for depth in "${values[@]}"
do
  echo $batch_name
  echo $depth
  export P P_test n_tasks T sigma depth fixed_w \
  script_name batch_name trial_ind lambda_val dataset permutation n_epochs

  for seed in "${seeds[@]}"
  do
    export seed
    sbatch --account=cox_lab --job-name=$batch_name --mem=$MEM_REQUEST -t $TIME_REQUEST -p $PARTITION\
    -o /n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/run_message_${trial_ind}.txt\
    /n/home11/haozheshan/ContinualLearning2022/cluster_scripts/multihead.sbatch.sh

    trial_ind=$((trial_ind+1))
    sleep 0.5
  done
done
