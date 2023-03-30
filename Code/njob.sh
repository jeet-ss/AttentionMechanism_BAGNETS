#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:59:00
#SBATCH --job-name=jss_0to1
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

module load python/3.9-anaconda
cd $HPCVAULT/Code
source myvenv/bin/activate
#echo "the args are $*"
#srun python FeatureExtractor.py
#srun python ReferenceChecker.py
#srun python NewAttentionModel.py $*
srun python SaliencyMap.py

