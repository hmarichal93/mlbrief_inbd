#!/bin/bash
DOCKERFILE=.ipol/Dockerfile
DOCKERIMAGE=inbd:v1
CONTAINERNAME=inbd_test

# Function to build the Docker image
build_image() {
    echo "Building Docker image..."
    docker build -f $DOCKERFILE . -t $DOCKERIMAGE
    echo "Docker image built successfully."
}

# Function to run the Docker image
run_image() {
    echo "Running Docker image..."
    docker run --name $CONTAINERNAME -it $DOCKERIMAGE /bin/bash
    echo "Docker image is running."
}

# Function to delete the Docker image
delete_image() {
    echo "Deleting Docker image..."
    docker rm -f $CONTAINERNAME
    docker rmi -f $DOCKERIMAGE
    echo "Docker image deleted."
}

# Function to copy file from host to container
copy_file() {
    echo "Copying file from host to container..."
    docker cp $1 $CONTAINERNAME:$2
    echo "File copied successfully."
}

# Function to get

# Check the script parameter
if [ "$1" == "build" ]; then
    build_image
elif [ "$1" == "run" ]; then
    run_image
elif [ "$1" == "delete" ]; then
    delete_image
else
    echo "Invalid parameter. Please use 'build' to build the image or 'run' to run the image."
fi