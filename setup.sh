#!/bin/bash

# ============================================================================
# Automatic Setup Script - RAG with Ollama
# ============================================================================
# This script automates the installation and configuration of the environment
# for the RAG (Retrieval-Augmented Generation) project using Ollama
# ============================================================================

set -e  # Stop script on error

# Output colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# ============================================================================
# STEP 1: Verify working directory
# ============================================================================
print_step "Verifying working directory..."

if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found. Are you in the correct directory?"
    exit 1
fi

print_success "Correct directory"

# ============================================================================
# STEP 2: Activate virtual environment
# ============================================================================
print_step "Activating virtual environment..."

if [ ! -d ".venv" ]; then
    print_error ".venv virtual environment not found"
    print_warning "First run: python -m venv .venv"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

if [ -z "$VIRTUAL_ENV" ]; then
    print_error "Could not activate virtual environment"
    exit 1
fi

print_success "Virtual environment activated: $VIRTUAL_ENV"

# ============================================================================
# STEP 3: Upgrade pip
# ============================================================================
print_step "Upgrading pip..."

pip install --upgrade pip --quiet

print_success "pip upgraded"

# ============================================================================
# STEP 4: Install Python dependencies
# ============================================================================
print_step "Installing Python dependencies from requirements.txt..."
print_warning "This may take several minutes..."

pip install -r requirements.txt

print_success "Python dependencies installed"

# ============================================================================
# STEP 5: Verify if Ollama is already installed
# ============================================================================
print_step "Verifying Ollama installation..."

if command -v ollama &> /dev/null; then
    print_success "Ollama is already installed ($(ollama --version))"
    OLLAMA_INSTALLED=true
else
    print_warning "Ollama is not installed"
    OLLAMA_INSTALLED=false
fi

# ============================================================================
# STEP 6: Install Ollama (if not installed)
# ============================================================================
if [ "$OLLAMA_INSTALLED" = false ]; then
    print_step "Installing Ollama..."
    print_warning "This may require sudo permissions"
    
    curl -fsSL https://ollama.com/install.sh | sh
    
    if command -v ollama &> /dev/null; then
        print_success "Ollama installed successfully ($(ollama --version))"
    else
        print_error "Ollama installation failed"
        exit 1
    fi
fi

# ============================================================================
# STEP 7: Start Ollama service (in background)
# ============================================================================
print_step "Verifying Ollama service..."

# Verify if Ollama is already running
if pgrep -x "ollama" > /dev/null; then
    print_success "Ollama service is already running"
else
    print_step "Starting Ollama service in background..."
    ollama serve > /dev/null 2>&1 &
    sleep 3  # Wait for service to start
    print_success "Ollama service started"
fi

# ============================================================================
# STEP 8: Download LLM model
# ============================================================================
print_step "Verifying model qwen2.5:7b..."

# Verify if model is already downloaded
if ollama list | grep -q "qwen2.5:7b"; then
    print_success "Model qwen2.5:7b is already downloaded"
else
    print_step "Downloading model qwen2.5:7b..."
    print_warning "This may take 5-10 minutes depending on your connection"
    
    ollama pull qwen2.5:7b
    
    print_success "Model qwen2.5:7b downloaded"
fi

# ============================================================================
# STEP 9: Verify everything is working
# ============================================================================
print_step "Verifying that the model responds correctly..."

# Test the model with a simple prompt
RESPONSE=$(echo "Hello, respond only with 'OK' if you understand me" | ollama run qwen2.5:7b 2>/dev/null)

if [ -n "$RESPONSE" ]; then
    print_success "Model working correctly"
    echo -e "${GREEN}Model response:${NC} $RESPONSE"
else
    print_error "Model did not respond correctly"
    exit 1
fi


# ============================================================================
# STEP 11: Final verification
# ============================================================================
echo ""
echo "============================================================================"
print_success "Setup completed successfully!"
echo "============================================================================"
echo ""
echo "Installation summary:"
echo "  • Virtual environment: $VIRTUAL_ENV"
echo "  • Ollama: $(ollama --version)"
echo "  • Model: qwen2.5:7b"
echo ""
echo "Next steps:"
echo "  1. The virtual environment is already activated"
echo "  2. You can test the model with: ollama run qwen2.5:7b"
echo "  3. Continue with the development of the RAG project"
echo ""
echo "To deactivate the virtual environment: deactivate"
echo "============================================================================"
