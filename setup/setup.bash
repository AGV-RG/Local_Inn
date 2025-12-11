#!/bin/bash

VENV_DIR="local-inn"

if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
fi

source $VENV_DIR/bin/activate

REQ_FILE="setup/requirements.txt"

if [ -f "$REQ_FILE" ]; then
    pip install --upgrade pip
    pip install -r "$REQ_FILE"
else
    echo "Warning: $REQ_FILE not found."
fi

if [ ! -d "data" ]; then
    mkdir data
fi