#!/bin/bash
# Quick setup script

echo "Setting up Spine Compensation LLM Bias Study..."

# Create results directory
mkdir -p results

# Copy .env if doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ Created .env file - please add your API keys"
else
    echo "✓ .env already exists"
fi

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys"
echo "2. Run: python run_experiment.py --test"
echo ""

