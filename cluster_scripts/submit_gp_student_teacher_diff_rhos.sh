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
# xsim=100  # ENTER PERCENTAGE VALUE (e.g. 80 for 0.80)
depth=1
lambda_val=100000
use_large_lambda_limit=1  # this ignores lambda_val and assumes inf lambda

context_strength=1.0
change_w_in_teachers=0.0
NSEEDS=100


MEM_REQUEST=5000 # memory requested, in MB
TIME_REQUEST=0-0:30
PARTITION=shared

batch_name=gp_toy_${depth}L_${n_tasks}x${P}_tsim${tsim}_NC${NC}_accumulation

directory="/n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/"
rm -rf $directory  # remove the directory
mkdir $directory

values=$(seq 0 10 100)

trial_ind=0

for xsim in ${values[@]}
do
  # echo $tsim
  echo $xsim
  export P P_test n_tasks T sigma N0 Nh NC radius tsim xsim N0context \
  depth seed lambda_val script_name trial_ind batch_name NSEEDS \
  context_strength change_w_in_teachers use_large_lambda_limit

  sbatch --account=cox_lab --job-name=$batch_name --mem=$MEM_REQUEST -t $TIME_REQUEST -p $PARTITION\
  -o /n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/run_message_${trial_ind}.txt\
  /n/home11/haozheshan/ContinualLearning2022/cluster_scripts/gp_student_teacher.sbatch.sh

  trial_ind=$((trial_ind+1))
  sleep 0.1
done
