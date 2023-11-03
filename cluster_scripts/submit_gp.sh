### MAY 23: USING LARGE LAMBDA APPROXIMATIONS IN gp.py

# script_name=gp.py
# dataset='mnist'
# task_type='permuted'
# P=500
# P_test=50  # for cifar100 this can be at most 200
# n_tasks=100
# T=0
# sigma=0.2
# resample=0
# permutation=1.0  # this argument is ignored for split tasks
# lambda_val=100000
# naive_gp=0
# save_outputs=0
# N0context=0
# context_strength=1.0
# depth=1

# MEM_REQUEST=8000 # memory requested, in MB
# TIME_REQUEST=0-12:30
# PARTITION=shared
# # batch_name=gp_${n_tasks}x${P}_${dataset}_${task_type}_${N0context}context_${depth}L_diff_strength
# batch_name=gp_${n_tasks}x${P}_${dataset}_${task_type}_diff_depth_double_check
# #batch_name=cifar_debug2


# directory="/n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/"
# rm -rf $directory  # remove the directory
# mkdir $directory

# # values=(1 2 3 4 5 6 7 8 9 10)
# # values=(0 25 50 75 100 125 150 175 200 225 250 275 300 325 350 375 400 425 450 475 500)
# values=(1 3 5 7 9)
# seeds=$(seq 1 1 100)
# trial_ind=0

# for depth in ${values[@]}
# do
#  echo $depth
#  export P P_test n_tasks T sigma depth permutation\
#  script_name batch_name trial_ind lambda_val task_type dataset permutation resample naive_gp save_outputs N0context context_strength

#  for seed in $seeds
#  do
#    export seed
#    sbatch --account=cox_lab --job-name=$batch_name --mem=$MEM_REQUEST -t $TIME_REQUEST -p $PARTITION\
#    -o /n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/run_message_${trial_ind}.txt\
#    /n/home11/haozheshan/ContinualLearning2022/cluster_scripts/gp.sbatch.sh

#    trial_ind=$((trial_ind+1))
#    sleep 0.01
#  done
# done

# !!!!!! SAVING NN OUTPUTS. Check gp.py (MAY 2, 2023)

script_name=gp.py
dataset='emnist'
task_type='permuted'
P=500
P_test=500 #default 500
n_tasks=20
T=0
sigma=0.2
#depth=1
resample=0
permutation=0.05
depth=1
naive_gp=0
save_outputs=1
N0context=0
context_strength=1.0


MEM_REQUEST=10000 # memory requested, in MB
TIME_REQUEST=0-3:30
PARTITION=shared
batch_name=gp_${n_tasks}x${P}_${dataset}_${task_type}_${depth}L_diff_lambda_outputs_upto1e6
#batch_name=cifar_debug2


directory="/n/home11/haozheshan/ContinualLearning2022/outputs/${batch_name}/"
rm -rf $directory  # remove the directory
mkdir $directory


# values=(10 13.7 18.9 25.9 35.6 48.9 67.2 92.4 127 174 239 329 452 621 853 1172 1610 2212 3039 4175 5736 7880 10826 14873 20433 28072 38566 52983 72789 100000)
values=(100 114 131 149 171 195 223 255 291 332 380 434 496 567 648 741 846 \
967 1105 1263 1444 1650 1885 2154 2462 2814 3215 3675 4199 4799 5484 6268 7163 \
8185 9354 10690 12217 13961 15955 18233 20837 23813 27213 31100 35541 40616 \
46416 53044 60619 69276 79168 90474 103393 118158 135031 154314 176351 201534 \
230313 263202 300788 343741 392828 448925 513033 586295 670019 765699 875042 \
1000000)

# values=(1 )
#values=(1 2 3 4 5 6 7 8 9 10)
#seeds=$(seq 0 1 19)  # first seed HAS to be 0 because the default data seed for SGD sims is 0
seeds=$(seq 0 1 1)
#values="$(seq 0 0.1 1.57)"
trial_ind=0
# values=(400 800 1600 3200)


# shellcheck disable=SC2068

for lambda_val in ${values[@]}
do
  echo $batch_name
#  echo $P
  export P P_test n_tasks T sigma depth permutation\
  script_name batch_name trial_ind lambda_val task_type dataset resample naive_gp save_outputs N0context context_strength

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