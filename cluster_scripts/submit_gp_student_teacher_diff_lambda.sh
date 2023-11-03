script_name=gp_student_teacher.py
P=50
P_test=100 #default 500
n_tasks=11
T=0
sigma=0.2
N0=100
N0context=0
Nh=100
NC=1
radius=0.1
tsim=100  # ENTER PERCENTAGE VALUE (e.g. 80 for 0.80)
xsim=75  # ENTER PERCENTAGE VALUE (e.g. 80 for 0.80)
depth=1
lambda_val=100000
context_strength=1.0
change_w_in_teachers=0.0
use_large_lambda_limit=0
input_share_variability=0  # this needs to be 0 if testing generalization
NSEEDS=30



MEM_REQUEST=5000 # memory requested, in MB
TIME_REQUEST=0-0:30
PARTITION=shared

batch_name=gp_toy_${n_tasks}x${P}_tsim${tsim}_xsim${xsim}_NC${NC}_diff_lambda

directory="/n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/"
rm -rf $directory  # remove the directory
mkdir $directory

# values=(1 4 7 10)
# values=(1 2 3 4 5 6 7 8 9 10)
# values=$(seq 0 10 100)
values=(0.10 0.23 0.55 1.27 2.98 6.95 16.24 37.93 88.59 206.91 483.29 1128.84 \
2636.65 6158.48 14384.50 33598.18 78476.00 183298.07 428133.24 1000000.00)
# values="$(seq 0 0.1 1.57)"

# values=(400 800 1600 3200)

trial_ind=0

for lambda_val in ${values[@]}
do
  echo $lambda_val
  export P P_test n_tasks T sigma N0 Nh NC radius tsim xsim N0context \
  depth seed lambda_val script_name trial_ind batch_name NSEEDS \
  context_strength change_w_in_teachers use_large_lambda_limit \
  input_share_variability

  sbatch --account=cox_lab --job-name=$batch_name --mem=$MEM_REQUEST -t $TIME_REQUEST -p $PARTITION\
  -o /n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/run_message_${trial_ind}.txt\
  /n/home11/haozheshan/ContinualLearning2022/cluster_scripts/gp_student_teacher.sbatch.sh

  trial_ind=$((trial_ind+1))
  sleep 0.1
done