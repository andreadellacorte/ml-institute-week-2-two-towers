#bin/bash

if [[ ! -d ".venv" ]]; then
    python -m venv .venv
    source .venv/bin/activate
else
    echo ".venv already exists. Skipping virtual environment creation."
    source .venv/bin/activate
fi

pip install --upgrade pip

if [[ "$1" == "--gpu" ]]; then
    echo "GPU mode selected. Installing GPU-specific dependencies..."
    pip install -r requirements-gpu.txt
elif [[ "$1" == "--cpu" ]]; then
    echo "CPU mode selected. Installing CPU-specific dependencies..."
    pip install -r requirements-cpu.txt
else
    echo "Invalid parameter. Please use '--gpu' for GPU dependencies or '--cpu' for CPU dependencies."
    exit 1
fi

wandb login

# if virtual environment is not active, provide instructions
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Virtual environment is not active. Please activate it using:"
    echo "source .venv/bin/activate"
else
    echo "Virtual environment is active and ready."
fi