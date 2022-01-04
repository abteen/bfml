#!/bin/bash

#Run with 'sbatch --export=ALL,seed=42 finetune_xlmr_large.slurm'

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32g
#SBATCH --time=7-00:00:00
#SBATCH --partition=blanca-kann
#SBATCH --qos=blanca-kann
#SBATCH --gres=gpu:v100:2



module purge
module load python

cd /projects/abeb4417/

source startup.sh

cd americasnli/

echo "seed is ${seed}"

for lang in bzd
do
./scripts/finetuning/finetune_mlm_es_xlmr_large.sh ${lang} ${seed}
done



