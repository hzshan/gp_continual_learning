
# Always check flags in the gradient.py file. 

# The `DIFF_DATA_SEED` flag controls whether different runs use different
# seeds to sample data.


script_name=gradient.py
P=2000
P_test=50
n_tasks=15
dataset='mnist'
task_type='permuted'
depth=1
manipulation_ratio=1.0
resample=0

N=2000
T=0
sigma=1.0
eta=0.1
minibatch=500
decay=0.0
n_epochs=1
n_steps=500000
l2=0.0001
whiten=1


MEM_REQUEST=10000 # memory requested, in MB
TIME_REQUEST=0-2:00
PARTITION=seas_gpu

batch_name=gradient_${n_tasks}x${P}_${dataset}_${task_type}_full_${depth}L_N${N}
# batch_name=${n_tasks}x${P}_${dataset}_${task_type}_full


directory="/n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/"
rm -rf $directory  # remove the directory
mkdir $directory

#values=(1 100 500 1000 5000 10000 50000 100000)
#values=(1000000)
#values=(1)
seeds=$(seq 1 1 10)
#values="$(seq 0 0.1 1.57)"
trial_ind=0
# values=(400 800 1600 3200)


# shellcheck disable=SC2068
echo $batch_name
echo $lambda_val
export P P_test n_tasks T sigma depth \
script_name batch_name trial_ind lambda_val dataset manipulation_ratio n_epochs l2 N n_steps eta task_type decay minibatch resample whiten

for seed in $seeds
do
  export seed
  sbatch --account=cox_lab --job-name=$batch_name --mem=$MEM_REQUEST -t $TIME_REQUEST -p $PARTITION\
  -o /n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/run_message_${trial_ind}.txt\
  /n/home11/haozheshan/ContinualLearning2022/cluster_scripts/gradient_sim.sbatch.sh

  trial_ind=$((trial_ind+1))
  sleep 0.25
done
