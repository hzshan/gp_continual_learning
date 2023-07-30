script_name=gp_toy_model.py
P=50
P_test=100 #default 500
n_tasks=30
T=0
sigma=0.2
N0=100
N0context=0
Nh=100
NC=10
radius=0.1
tsim=60  # ENTER PERCENTAGE VALUE (e.g. 80 for 0.80)
xsim=90  # ENTER PERCENTAGE VALUE (e.g. 80 for 0.80)
depth=1
lambda_val=100000
context_strength=1.0

NSEEDS=100


MEM_REQUEST=5000 # memory requested, in MB
TIME_REQUEST=0-6:00
PARTITION=shared


batch_name=gp_toy_${n_tasks}x${P}_xsim${xsim}_${depth}L_diff_tsim

directory="/n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/"
rm -rf $directory  # remove the directory
mkdir $directory

# values=(1 4 7 10)
# values=(1 2 3 4 5 6 7 8 9 10)
values=$(seq 0 10 100)
# values=(10 30 100 300 1000 3000 10000 30000 100000)
# values="$(seq 0 0.1 1.57)"

# values=(400 800 1600 3200)

trial_ind=0

for tsim in ${values[@]}
do
  
  echo $tsim
  export P P_test n_tasks T sigma N0 Nh NC radius tsim xsim N0context depth seed lambda_val script_name trial_ind batch_name NSEEDS context_strength

  sbatch --account=cox_lab --job-name=$batch_name --mem=$MEM_REQUEST -t $TIME_REQUEST -p $PARTITION\
  -o /n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/run_message_${trial_ind}.txt\
  /n/home11/haozheshan/ContinualLearning2022/cluster_scripts/gp_toy.sbatch.sh

  trial_ind=$((trial_ind+1))
  sleep 0.1
done


