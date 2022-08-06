script_name=sgd.py
P=100
P_test=1000
N=1000
n_tasks=5
T=0
sigma=0.2
depth=1
eta=0.1
permutation=1.0
dataset='mnist'
n_epochs=1
n_steps=1000000


MEM_REQUEST=4000 # memory requested, in MB
TIME_REQUEST=0-0:20
PARTITION=seas_gpu_requeue

batch_name=Langevin_${n_tasks}x${P}_${dataset}_${depth}L_${eta}_N${N}
#batch_name=cifar_debug2


directory="/n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/"
rm -rf $directory  # remove the directory
mkdir $directory

values=$(seq 0 0.0002 0.0004)
#values=(1 100 500 1000 5000 10000 50000 100000)
#values=(1000000)
#values=(1)
#seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
#seeds=(0 1 2 3 4 5 6 7)
seeds=$(seq 1 1 100)
#values="$(seq 0 0.1 1.57)"
trial_ind=0
# values=(400 800 1600 3200)


# shellcheck disable=SC2068

for l2 in ${values[@]}
do
  echo $batch_name
  echo $lambda_val
  export P P_test n_tasks T sigma depth \
  script_name batch_name trial_ind lambda_val dataset permutation n_epochs l2 N n_steps eta

  for seed in ${seeds[@]}
  do
    export seed
    sbatch --account=cox_lab --job-name=$batch_name --mem=$MEM_REQUEST -t $TIME_REQUEST -p $PARTITION\
    -o /n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/run_message_${trial_ind}.txt\
    /n/home11/haozheshan/ContinualLearning2022/cluster_scripts/sgd.sbatch.sh

    trial_ind=$((trial_ind+1))
    sleep 0.25
  done
done
