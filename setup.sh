#bin/bash

# Save the value of $VIRTUAL_ENV at the top
CURRENT_VIRTUAL_ENV="$VIRTUAL_ENV"
echo "Current VIRTUAL_ENV: $CURRENT_VIRTUAL_ENV"

if [[ ! -d ".venv" ]]; then
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
else
    echo ".venv already exists. Skipping virtual environment creation."
fi

source .venv/bin/activate
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

# Install gh CLI if not installed
if ! command -v gh &> /dev/null; then
    echo "gh CLI not found. Installing..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        (type -p wget >/dev/null || (apt update && apt-get install wget -y)) \
            && mkdir -p -m 755 /etc/apt/keyrings \
                && out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
                && cat $out | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
            && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
            && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
            && apt update \
            && apt install gh -y
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install gh
    else
        echo "Unsupported OS. Please install gh CLI manually."
    fi
else
    echo "gh CLI is already installed."
fi

if ! gh auth status &> /dev/null; then
    echo "gh CLI is not logged in. Logging in..."
    gh auth login
else
    echo "gh CLI is already logged in as $(gh auth status | grep 'Logged in to' | awk '{print $3}')"
fi

# if virtual environment is not active, provide instructions
if [[ "$CURRENT_VIRTUAL_ENV" == "" ]]; then
    echo "Virtual environment is not active. Please activate it using:"
    echo "source .venv/bin/activate"
else
    echo "Virtual environment is active and ready."
fi