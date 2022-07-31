script_name=sgd.py
P=1000
P_test=500
N=2000
n_tasks=2
T=0
sigma=0.2
depth=3
eta=1.0
permutation=1.0
dataset='fashion'
fixed_w=0
n_epochs=1
n_steps=2000


MEM_REQUEST=4000 # memory requested, in MB
TIME_REQUEST=0-2:20
PARTITION=shared

batch_name=SGD_${n_tasks}x${P}_${dataset}_${depth}L_permute${permutation}_short
#batch_name=cifar_debug2


directory="/n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/"
rm -rf $directory  # remove the directory
mkdir $directory

values=$(seq 0 0.02 0.3)
#values=(1 100 500 1000 5000 10000 50000 100000)
#values=(1000000)
#values=(1)
#seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
seeds=(0 1 2 3 4 5 6 7)
#values="$(seq 0 0.1 1.57)"
trial_ind=0
# values=(400 800 1600 3200)


# shellcheck disable=SC2068

for l2 in ${values[@]}
do
  echo $batch_name
  echo $lambda_val
  export P P_test n_tasks T sigma depth fixed_w \
  script_name batch_name trial_ind lambda_val dataset permutation n_epochs l2 N n_steps eta

  for seed in "${seeds[@]}"
  do
    export seed
    sbatch --account=cox_lab --job-name=$batch_name --mem=$MEM_REQUEST -t $TIME_REQUEST -p $PARTITION\
    -o /n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/run_message_${trial_ind}.txt\
    /n/home11/haozheshan/ContinualLearning2022/cluster_scripts/sgd.sbatch.sh

    trial_ind=$((trial_ind+1))
    sleep 0.25
  done
done
