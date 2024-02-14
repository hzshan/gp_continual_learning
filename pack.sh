module load python
source activate continual
cd /n/home11/haozheshan/ContinualLearning2022 || exit

python pack_data.py --batch_to_pack $1

# Utility script for combining individual .results files into a single .packed_results file.
# The main advantage is that this makes file transfer much faster
# Example usage:
# sh pack.sh gp_toy_15x100_1L_radius0.5_diff_tsim_and_xsim_anterograde