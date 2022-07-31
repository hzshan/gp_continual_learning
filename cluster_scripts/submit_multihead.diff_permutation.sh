script_name=multihead.py
P=1000
P_test=1000
n_tasks=2
T=0
sigma=0.2
depth=1
dataset='mnist'
fixed_w=0
n_epochs=1
interpolate=0
resample=0


MEM_REQUEST=3000 # memory requested, in MB
TIME_REQUEST=0-0:20
PARTITION=shared

batch_name=${n_tasks}x${P}_${dataset}_${depth}L_resample${resample}_different_permutation_lambda1e5
#batch_name=cifar_debug2


directory="/n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/"
rm -rf $directory  # remove the directory
mkdir $directory

values="$(seq 0 0.05 1)"
#values=(1 100 500 1000 5000 10000 50000 100000)
#values=(1000000)
#values=(1)
seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
#seeds=(0 1 2 3 4)
#values="$(seq 0 0.1 1.57)"
trial_ind=0
# values=(400 800 1600 3200)

lambda_val=1e5
for permutation in $values
do
  echo $batch_name
  echo $permutation
  export P P_test n_tasks T sigma depth fixed_w \
  script_name batch_name trial_ind lambda_val dataset permutation n_epochs interpolate resample

  for seed in "${seeds[@]}"
  do
    export seed
    sbatch --account=cox_lab --job-name=$batch_name --mem=$MEM_REQUEST -t $TIME_REQUEST -p $PARTITION\
    -o /n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/run_message_${trial_ind}.txt\
    /n/home11/haozheshan/ContinualLearning2022/cluster_scripts/multihead.sbatch.sh

    trial_ind=$((trial_ind+1))
    sleep 0.1
  done
done
