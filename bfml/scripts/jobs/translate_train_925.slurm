#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32g
#SBATCH --time=7-00:00:00
#SBATCH --partition=blanca-kann
#SBATCH --qos=blanca-kann
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/projects/abeb4417/outputs/americasnli/translate_train_925.out


module purge
module load python

cd /projects/abeb4417/

source startup.sh

cd americasnli/

for lang in nah	oto	quy	shp	tar
do
./scripts/finetuning/finetune_translate_train.sh ${lang} 925
done
