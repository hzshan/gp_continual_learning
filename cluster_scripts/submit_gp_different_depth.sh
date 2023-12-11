script_name=gp.py
dataset='mnist'
task_type='permuted'
P=500
P_test=100  # for cifar100 this can be at most 200
n_tasks=100
T=0
sigma=0.2
resample=0
permutation=1.0  # this argument is ignored for split tasks
lambda_val=1000000
use_large_lambda_limit=1  # if 1, ignore lambda_val and assume large lambda
naive_gp=0
save_outputs=0
N0context=0
context_strength=1.0
depth=1
whiten=1

MEM_REQUEST=8000 # memory requested, in MB
TIME_REQUEST=0-1:30
PARTITION=shared
# batch_name=gp_${n_tasks}x${P}_${dataset}_${task_type}_${N0context}context_${depth}L_diff_strength
batch_name=gp_${n_tasks}x${P}_${dataset}_${task_type}_diff_depth_whiten
#batch_name=cifar_debug2


directory="/n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/"
rm -rf $directory  # remove the directory
mkdir $directory

# values=(1 2 3 4 5 6 7 8 9 10)
# values=(0 25 50 75 100 125 150 175 200 225 250 275 300 325 350 375 400 425 450 475 500)
values=(1 3 5 7 9)
seeds=$(seq 1 1 100)
trial_ind=0

for depth in ${values[@]}
do
 echo $depth
 export P P_test n_tasks T sigma depth permutation\
 script_name batch_name trial_ind lambda_val task_type dataset permutation\
  resample naive_gp save_outputs N0context context_strength\
   use_large_lambda_limit whiten

 for seed in $seeds
 do
   export seed
   sbatch --account=cox_lab --job-name=$batch_name --mem=$MEM_REQUEST -t $TIME_REQUEST -p $PARTITION\
   -o /n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/run_message_${trial_ind}.txt\
   /n/home11/haozheshan/ContinualLearning2022/cluster_scripts/gp.sbatch.sh

   trial_ind=$((trial_ind+1))
   sleep 0.01
 done
done