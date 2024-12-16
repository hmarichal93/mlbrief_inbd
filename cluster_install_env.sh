#!/bin/bash
#SBATCH --job-name=inbd
#SBATCH --ntasks=16
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --tmp=100G
#SBATCH --mail-user=henry.marichal@fing.edu.uy

# de acuerdo a lo que quiera ejecutar puede elegir entre las siguientes tres líneas.
#SBATCH --gres=gpu:1 # se solicita una gpu cualquiera( va a tomar la primera que quede disponible indistintamente si es una p100 o una a100)


#SBATCH --partition=normal
#SBATCH --qos=gpu


source /etc/profile.d/modules.sh
source /clusteruy/home/henry.marichal/miniconda3/etc/profile.d/conda.sh
conda activate inbd_gpu

cd /clusteruy/home/henry.marichal/repos/mlbrief_inbd/mlbrief_inbd/INBD
pip install -r requirements.txt