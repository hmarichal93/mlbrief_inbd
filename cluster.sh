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

# -------------------------------------------------------
#disco local SSD local al nodo. /clusteruy/home/henry.marichal se accede via NFS (puede ser realmente lento)
#el espacio local a utilizar se reserva dcon --tmp=XXXGb
ROOT_DIR=$2
HOME_DATASET_DIR=$3
HOME_RESULTADOS_DIR=$4
SEGMENTATION_MODEL=$5
IMAGES_FILE=$6
ANNOTATIONS_FILE=$7
LOCAL_NODE_DIR=/scratch/henry.marichal/
HOME_RESULTADOS_MODEL_DIR=$HOME_RESULTADOS_DIR/model

# -------------------------------------------------------
#other variables
#NODE_RESULTADOS_DIR=$LOCAL_NODE_DIR/inbd/resultados
#NODE_DATASET_DIR=$LOCAL_NODE_DIR/inbd/EH

#NODE_MODEL_RESULTADOS_DIR=$NODE_RESULTADOS_DIR/model
stdout_file="$HOME_RESULTADOS_DIR/stdout.txt"
stderr_file="$HOME_RESULTADOS_DIR/stderr.txt"
# Define a function to check the result of a command
check_command_result() {
    # Run the command passed as an argument
    "$@"

    # Check the exit status
    if [ $? -eq 0 ]; then
        echo "Command was successful."
    else
        echo "Command failed with an error."
        exit 1
    fi
}

####Prepare directories
#rm -rf $NODE_DATASET_DIR
#rm -rf $NODE_RESULTADOS_DIR
#rm -rf $HOME_RESULTADOS_DIR

#check_command_result mkdir -p $NODE_DATASET_DIR
#check_command_result mkdir -p $NODE_RESULTADOS_DIR
check_command_result mkdir -p $HOME_RESULTADOS_DIR

####Move dataset to node local disk
#check_command_result cp  -r $HOME_DATASET_DIR $NODE_DATASET_DIR


# -------------------------------------------------------
# Run the program
cd $ROOT_DIR

if [ "$1" == "segmentation" ]; then
    echo "Segmentation"
    python main.py train segmentation $HOME_DATASET_DIR/$IMAGES_FILE $HOME_DATASET_DIR/$ANNOTATIONS_FILE \
    --output $HOME_RESULTADOS_MODEL_DIR --epochs 100 --downsample 1 --transfer_learning --model_path $SEGMENTATION_MODEL > "$stdout_file" 2> "$stderr_file"
fi
if [ "$1" == "INBD" ]; then
    echo "INBD"
    INBD_MODEL=$8
    python main.py train INBD $HOME_DATASET_DIR/$6 $HOME_DATASET_DIR/$7 \
           --segmentationmodel=$SEGMENTATION_MODEL --downsample 1 --output $HOME_RESULTADOS_MODEL_DIR --epochs 100\
           --transfer_learning --model_path $INBD_MODEL > "$stdout_file" 2> "$stderr_file"
fi


# -------------------------------------------------------
#copy results to HOME
mkdir -p $HOME_RESULTADOS_DIR
#cp -r $NODE_RESULTADOS_DIR/* $HOME_RESULTADOS_DIR
#cp -r $NODE_DATASET_DIR/* $HOME_RESULTADOS_DIR
#delete temporal files
#rm -rf $NODE_RESULTADOS_DIR
#rm -rf $NODE_DATASET_DIR