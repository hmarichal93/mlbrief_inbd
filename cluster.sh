#!/bin/bash
#SBATCH --job-name=inbd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --tmp=100G
#SBATCH --mail-user=henry.marichal@fing.edu.uy
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=normal
#SBATCH --qos=gpu



# Cargar módulos y activar el entorno
source /etc/profile.d/modules.sh
source /clusteruy/home/henry.marichal/miniconda3/etc/profile.d/conda.sh
conda activate deep_cstrd

# Variables
ROOT_DIR=$1
HOME_DATASET_DIR=$2
HOME_RESULTADOS_DIR=$3
DATASET_NAME=$4
NODE_DATASET_DIR=/scratch/henry.marichal/
NODE_RESULTADOS_DIR=$NODE_DATASET_DIR/results
EPOCHS=$5

# Función para verificar el resultado de un comando
check_command_result() {
    "$@"
    if [ $? -ne 0 ]; then
        echo "Error: El comando falló."
        exit 1
    fi
}

# Preparar directorios
check_command_result rm -rf $NODE_DATASET_DIR $NODE_RESULTADOS_DIR
check_command_result mkdir -p $NODE_DATASET_DIR $NODE_RESULTADOS_DIR

# Copiar dataset al disco local del nodo
check_command_result cp -r $HOME_DATASET_DIR $NODE_DATASET_DIR

# Entrenar el modelo
cd $ROOT_DIR

for i in {1..5}; do
  MODEL_DIR="$NODE_RESULTADOS_DIR/model_$i"
  check_command_result mkdir -p $MODEL_DIR
  python main.py train segmentation $NODE_DATASET_DIR/$DATASET_NAME/$IMAGES_FILE $NODE_DATASET_DIR/$DATASET_NAME/$ANNOTATIONS_FILE \
      --output $MODEL_DIR --epochs $EPOCHS --downsample 1

  SEGMENTATION_MODEL=$(find ${MODEL_DIR} -name model.pt.zip)

  python main.py train INBD $NODE_DATASET_DIR/$DATASET_NAME/$IMAGES_FILE $NODE_DATASET_DIR/$DATASET_NAME/$ANNOTATIONS_FILE \
         --segmentationmodel=$SEGMENTATION_MODEL --downsample 1 --output $MODEL_DIR --epochs $EPOCHS

  HOME_ITERATION_DIR="$HOME_RESULTADOS_DIR/model_$i"
  check_command_result mkdir -p $HOME_ITERATION_DIR
  check_command_result cp -r $MODEL_DIR $HOME_ITERATION_DIR

done


# -------------------------------------------------------
#copy results to HOME
mkdir -p $HOME_RESULTADOS_DIR
#cp -r $NODE_RESULTADOS_DIR/* $HOME_RESULTADOS_DIR
#cp -r $NODE_DATASET_DIR/* $HOME_RESULTADOS_DIR
#delete temporal files
#rm -rf $NODE_RESULTADOS_DIR
#rm -rf $NODE_DATASET_DIR