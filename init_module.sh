#!/bin/bash
BASE_MODULE_NAME="BaseCppFiles"
MODULE_NAME=$1
echo "Initializing module: $MODULE_NAME"
if [ -d "$MODULE_NAME" ]; then
    echo "Module $MODULE_NAME already exists"
    exit 1
fi
mkdir -p $MODULE_NAME

cp -f ./$BASE_MODULE_NAME/* ./$MODULE_NAME/