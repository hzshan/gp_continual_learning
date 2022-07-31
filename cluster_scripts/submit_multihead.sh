script_name=multihead.py
P=1000
P_test=1000
n_tasks=30
T=0
sigma=0.2
depth=3
permutation=1
dataset='mnist'
fixed_w=0
n_epochs=1
interpolate=0
resample=0


MEM_REQUEST=3000 # memory requested, in MB
TIME_REQUEST=0-30:30
PARTITION=shared

batch_name=${n_tasks}x${P}_${dataset}_${depth}L_fW${fixed_w}_permute${permutation}_resample${resample}
#batch_name=cifar_debug2


directory="/n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/"
rm -rf $directory  # remove the directory
mkdir $directory

values=(1 3 10 30 100 300 1000 3000 10000 30000 100000 300000 1000000 3000000 10000000)
#values=(1 100 1000 10000 10000000 100000000)
#values=(10000000)
#values=(1)
#seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
seeds=(0 1 2 3 4 5 6 7 8 9)
#seeds=(0 1 2 3 4)
#values="$(seq 0 0.1 1.57)"
trial_ind=0
# values=(400 800 1600 3200)


# shellcheck disable=SC2068
for lambda_val in "${values[@]}"
do
  echo $batch_name
  echo $lambda_val
  export P P_test n_tasks T sigma depth fixed_w \
  script_name batch_name trial_ind lambda_val dataset permutation n_epochs interpolate resample

  for seed in "${seeds[@]}"
  do
    export seed
    sbatch --account=cox_lab --job-name=$batch_name --mem=$MEM_REQUEST -t $TIME_REQUEST -p $PARTITION\
    -o /n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/run_message_${trial_ind}.txt\
    /n/home11/haozheshan/ContinualLearning2022/cluster_scripts/multihead.sbatch.sh

    trial_ind=$((trial_ind+1))
    sleep 0.25
  done
done
