#!/bin/bash
# filename          : launch_hope.sh
# description       : Script to run hope classification task on HPCC
# author            : Ebuka Ijezue
# email             : cijezue@ttu.edu
# date              : March 21, 2025
# version           : 1.0
# usage             : sbatch launch_hope.sh
# notes             : Designed for Quanah.hpcc.ttu.edu, runs BERT-based hope classification
#                   : Assumes working directory is /lustre/work/cijezue/Hope/
# license           : MIT License
#==============================================================================

#SBATCH --job-name=HopeClassifier
#SBATCH --output=out/%x.o%j
#SBATCH --error=out/%x.e%j
#SBATCH --partition=nocona
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=14:00:00          # 14 hours, adjust as needed
#SBATCH --mem-per-cpu=40GB    # 40GB per core (320GB total with 8 cores)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cijezue@ttu.edu

# Store original field separators
OIFS="$IFS"
IFS=$'\n'

# Load Conda environment from Miniconda3 installation
source /lustre/work/cijezue/miniconda3/etc/profile.d/conda.sh
conda activate hopeenv

# Ensure output directory exists in the working directory
mkdir -p /lustre/work/cijezue/Hope/out

# Run the Python script from the current directory
echo -e "\n\nRunning Hope Classification Task\n"
python /lustre/work/cijezue/Hope/hope_classifier.py > /lustre/work/cijezue/Hope/out/hope_output.log 2>&1

# Capture the last line of output (e.g., accuracies)
LAST_LINE=$(tail -n 1 /lustre/work/cijezue/Hope/out/hope_output.log)
echo "Last line of Python output: $LAST_LINE"

# Deactivate Conda and restore field separators
conda deactivate
IFS="$OIFS"

echo "Job completed."