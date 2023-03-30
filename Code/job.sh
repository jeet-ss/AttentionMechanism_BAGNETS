#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --job-name=jss_cle100
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

module load python
cd $HPCVAULT/Code
source .venv/bin/activate
#echo "the args are $*"
#srun python FeatureExtractor.py
#srun python ReferenceChecker.py
srun python NewAttentionModel.py $*
#srun python3 SaliencyMap.py
