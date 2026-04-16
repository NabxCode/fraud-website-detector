#!/bin/bash

# Fraud Website Detection - Runner Script
# This script starts the Flask API and/or Gradio interface

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
FLASK_PORT=5001
FLASK_ALT_PORT=5002
GRADIO_PORT=7860
GRADIO_ALT_PORT=7861
FLASK_PID=""
GRADIO_PID=""

# Set environment for XGBoost on macOS (libomp)
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [ -d "/opt/homebrew/opt/libomp" ]; then
        export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
    fi
fi

# Platform detection
PLATFORM="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    PLATFORM="windows_gitbash"
elif [[ -f /proc/version ]] && grep -q Microsoft /proc/version 2>/dev/null; then
    PLATFORM="windows_wsl"
fi

# Check Windows compatibility
if [[ "$PLATFORM" == "unknown" ]] && [[ "$OS" == "Windows_NT" ]]; then
    echo "ERROR: This script requires a Unix-like environment on Windows."
    echo ""
    echo "Please use one of the following:"
    echo "  1. WSL (Windows Subsystem for Linux) - Recommended"
    echo "     Install: wsl --install"
    echo ""
    echo "  2. Git Bash (comes with Git for Windows)"
    echo "     Download: https://git-scm.com/download/win"
    echo ""
    echo "  3. Manual Python commands:"
    echo "     python -m venv .venv"
    echo "     .venv\\Scripts\\activate"
    echo "     pip install -r requirements.txt"
    echo "     python src/app.py"
    exit 1
fi

# Platform-specific settings
case "$PLATFORM" in
    linux|macos)
        PYTHON_BIN="python3"
        VENV_PYTHON=".venv/bin/python"
        ;;
    windows_wsl)
        PYTHON_BIN="python3"
        VENV_PYTHON=".venv/bin/python"
        print_info "Detected WSL environment"
        ;;
    windows_gitbash)
        PYTHON_BIN="python"
        VENV_PYTHON=".venv/Scripts/python.exe"
        print_info "Detected Git Bash environment"
        ;;
esac

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a port is in use (cross-platform)
check_port() {
    local port=$1
    case "$PLATFORM" in
        windows_gitbash)
            if netstat -an 2>/dev/null | grep -q ":$port "; then
                return 0
            else
                return 1
            fi
            ;;
        *)
            if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
                return 0
            else
                return 1
            fi
            ;;
    esac
}

# Function to check if we can free a port (cross-platform)
kill_port() {
    local port=$1
    local pids=""
    
    case "$PLATFORM" in
        windows_gitbash)
            # Windows uses netstat and taskkill
            pids=$(netstat -ano 2>/dev/null | grep ":$port " | awk '{print $5}' | grep -v '0$' | sort -u)
            ;;
        *)
            pids=$(lsof -Pi :$port -sTCP:LISTEN -t 2>/dev/null)
            ;;
    esac
    
    if [ -n "$pids" ]; then
        print_warning "Found processes on port $port: $pids"
        for pid in $pids; do
            case "$PLATFORM" in
                windows_gitbash)
                    print_info "Killing process (PID: $pid)"
                    if taskkill //F //PID "$pid" 2>/dev/null; then
                        print_success "Killed process (PID: $pid)"
                    else
                        print_warning "Could not kill process (PID: $pid)"
                        return 1
                    fi
                    ;;
                *)
                    local proc_name=$(ps -p $pid -o comm= 2>/dev/null || echo "unknown")
                    print_info "Process: $proc_name (PID: $pid)"
                    # Check if it's a system process that we can't kill
                    if [[ "$proc_name" == *"ControlCe"* ]] || [[ "$proc_name" == *"airportd"* ]] || [[ "$proc_name" == "/System"* ]]; then
                        print_warning "Port $port is used by macOS system process: $proc_name"
                        return 1
                    fi
                    # Try to kill non-system processes
                    if kill $pid 2>/dev/null || kill -9 $pid 2>/dev/null; then
                        print_success "Killed $proc_name (PID: $pid)"
                    else
                        print_warning "Could not kill $proc_name (PID: $pid)"
                        return 1
                    fi
                    ;;
            esac
        done
        sleep 1
        # Verify port is now free
        if check_port $port; then
            print_error "Failed to free port $port"
            return 1
        fi
        print_success "Port $port is now free"
    fi
    return 0
}

# Function to check if our Flask is running on port
is_our_flask() {
    local port=$1
    if curl -s "http://localhost:$port/health" 2>/dev/null | grep -q "phishing"; then
        return 0
    fi
    return 1
}

# Function to find an available port starting from a base port
find_available_port() {
    local base_port=$1
    local max_attempts=10
    local port=$base_port
    
    for ((i=0; i<max_attempts; i++)); do
        if ! check_port $port; then
            echo $port
            return 0
        fi
        ((port++))
    done
    return 1
}

# Function to show log tail on failure
show_log_on_failure() {
    local log_file=$1
    local name=$2
    if [ -f "$log_file" ]; then
        print_info "Showing last 50 lines from $name logs:"
        echo "========================================"
        tail -n 50 "$log_file" 2>/dev/null || cat "$log_file"
        echo "========================================"
    fi
}

# Function to check process status and logs
debug_process_failure() {
    local pid=$1
    local log_file=$2
    local name=$3
    
    print_info "Debugging $name failure..."
    
    # Check if process exists
    if [ -n "$pid" ]; then
        if ps -p $pid > /dev/null 2>&1; then
            print_warning "Process $pid is still running but not responding"
        else
            print_error "Process $pid has exited"
            # Get exit code if available
            if [ -f "$log_file" ]; then
                print_info "Checking logs for exit reason..."
            fi
        fi
    fi
    
    # Show the log content
    show_log_on_failure "$log_file" "$name"
    
    # Check for common Python import errors
    if [ -f "$log_file" ]; then
        if grep -q "ModuleNotFoundError" "$log_file" 2>/dev/null; then
            print_error "Missing Python dependencies detected!"
            print_info "Auto-installing missing dependencies..."
            if install_dependencies; then
                print_success "Dependencies installed. Retrying..."
                # Set a flag to indicate we should retry
                AUTO_INSTALLED=1
            else
                print_error "Failed to auto-install dependencies"
                print_info "Try manually running: pip install -r requirements.txt"
            fi
        elif grep -q "Permission denied" "$log_file" 2>/dev/null; then
            print_error "Permission denied error detected!"
        elif grep -q "Address already in use" "$log_file" 2>/dev/null; then
            print_error "Port conflict detected!"
        fi
    fi
}

# Function to wait for a service to be ready
wait_for_service() {
    local url=$1
    local name=$2
    local log_file=$3
    local max_attempts=30
    local attempt=1
    
    print_info "Waiting for $name to start..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            print_success "$name is ready!"
            return 0
        fi
        
        # Check if process died early
        if [ -n "$FLASK_PID" ] && [ "$name" = "Flask API" ]; then
            if ! ps -p $FLASK_PID > /dev/null 2>&1; then
                echo ""
                print_error "$name process exited unexpectedly during startup"
                debug_process_failure "$FLASK_PID" "$log_file" "$name"
                return 1
            fi
        fi
        
        echo -n "."
        sleep 1
        ((attempt++))
    done
    
    echo ""
    print_error "$name failed to start within ${max_attempts} seconds"
    debug_process_failure "$FLASK_PID" "$log_file" "$name"
    return 1
}

# Function to cleanup processes on exit
cleanup() {
    print_info "Cleaning up..."
    
    if [ -n "$FLASK_PID" ]; then
        print_info "Stopping Flask API (PID: $FLASK_PID)..."
        kill $FLASK_PID 2>/dev/null || true
    fi
    
    if [ -n "$GRADIO_PID" ]; then
        print_info "Stopping Gradio app (PID: $GRADIO_PID)..."
        kill $GRADIO_PID 2>/dev/null || true
    fi
    
    print_success "Cleanup complete"
    exit 0
}

# Trap signals to cleanup
trap cleanup SIGINT SIGTERM EXIT

# Function to show usage
show_usage() {
    cat << EOF
Usage: ./run.sh [OPTION]

Options:
    flask       Start only the Flask API
    gradio      Start only the Gradio interface (requires Flask to be running)
    both        Start Flask API and then Gradio (default)
    api         Alias for 'flask'
    ui          Alias for 'gradio'
    all         Alias for 'both'
    -h, --help  Show this help message

Examples:
    ./run.sh              # Start both Flask and Gradio
    ./run.sh flask        # Start only Flask API
    ./run.sh gradio       # Start only Gradio (Flask must be running)
    ./run.sh both         # Start both services

Services:
    Flask API:  http://localhost:5001
    Gradio UI:  http://localhost:7860

EOF
}

# Function to install Python dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        print_error "Failed to install dependencies"
        return 1
    fi
    print_success "Dependencies installed successfully"
    return 0
}

# Function to create virtual environment
create_venv() {
    local venv_name=$1
    print_info "Creating virtual environment: $venv_name..."
    $PYTHON_BIN -m venv "$venv_name"
    if [ $? -ne 0 ]; then
        print_error "Failed to create virtual environment"
        return 1
    fi
    print_success "Virtual environment created: $venv_name"
    return 0
}

# Function to check Python environment
check_environment() {
    print_info "Checking environment..."
    
    # Check Python using platform-specific command
    if ! command -v $PYTHON_BIN &> /dev/null; then
        print_error "Python is not installed"
        exit 1
    fi
    
    PYTHON_CMD=$(command -v $PYTHON_BIN)
    print_success "Python found: $PYTHON_CMD"
    
    # Platform-specific venv paths
    local venv_activate=".venv/bin/activate"
    local venv_python=".venv/bin/python"
    if [[ "$PLATFORM" == "windows_gitbash" ]]; then
        venv_activate=".venv/Scripts/activate"
        venv_python=".venv/Scripts/python.exe"
    fi
    
    # Check if virtual environment exists, create if not
    if [ -d ".venv" ]; then
        print_info "Activating virtual environment (.venv)..."
        source "$venv_activate"
        # Update PYTHON_CMD to use venv Python
        PYTHON_CMD="$venv_python"
    elif [ -d "venv" ]; then
        print_info "Activating virtual environment (venv)..."
        if [[ "$PLATFORM" == "windows_gitbash" ]]; then
            source "venv/Scripts/activate"
            PYTHON_CMD="venv/Scripts/python.exe"
        else
            source "venv/bin/activate"
            PYTHON_CMD="venv/bin/python"
        fi
    else
        print_warning "No virtual environment found"
        read -p "Create virtual environment? (recommended) [Y/n]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            print_warning "Skipping virtual environment creation"
            print_info "Installing dependencies to system Python..."
        else
            if create_venv ".venv"; then
                source "$venv_activate"
                print_success "Virtual environment activated"
                # Update PYTHON_CMD to use venv Python
                PYTHON_CMD="$venv_python"
            else
                print_warning "Failed to create virtual environment, using system Python"
            fi
        fi
    fi
    
    # Show which Python will be used
    print_info "Using Python: $PYTHON_CMD"
    
    # Check required packages and install if missing
    if ! $PYTHON_CMD -c "import flask" 2>/dev/null; then
        print_warning "Flask not found. Installing requirements..."
        install_dependencies
    fi
    
    print_success "Environment check complete"
}

# Function to check if model is trained
check_model() {
    if [ ! -f "models/phishing_detector.pkl" ]; then
        print_warning "Trained model not found at models/phishing_detector.pkl"
        echo ""
        echo "Choose an option:"
        echo "  1) Create quick synthetic model (30 seconds, for demo)"
        echo "  2) Train on sample URLs (2-3 minutes, better quality)"
        echo "  3) Skip (API will not work without model)"
        echo ""
        read -p "Enter choice (1/2/3): " -n 1 -r
        echo
        
        case $REPLY in
            1)
                print_info "Creating quick synthetic model..."
                $PYTHON_CMD download_model.py --quick
                if [ $? -eq 0 ]; then
                    print_success "Quick model created successfully"
                else
                    print_error "Failed to create model"
                    exit 1
                fi
                ;;
            2)
                print_info "Training model on sample data..."
                $PYTHON_CMD train.py --sample --n-samples 200
                if [ $? -eq 0 ]; then
                    print_success "Model trained successfully"
                else
                    print_error "Model training failed"
                    exit 1
                fi
                ;;
            *)
                print_warning "Continuing without model. API will return errors."
                ;;
        esac
    else
        print_success "Model found: models/phishing_detector.pkl"
    fi
}

# Function to start Flask API
start_flask() {
    local use_port=$FLASK_PORT
    
    print_info "Starting Flask API on port $use_port..."
    
    # Check if port is already in use
    if check_port $use_port; then
        if is_our_flask $use_port; then
            print_success "Flask API is already running on port $use_port"
            return 0
        fi
        print_warning "Port $use_port is in use by another process"
        if ! kill_port $use_port; then
            print_info "Trying to find an available port..."
            local new_port=$(find_available_port $FLASK_ALT_PORT)
            if [ -z "$new_port" ]; then
                print_error "Cannot find an available port"
                return 1
            fi
            use_port=$new_port
            print_success "Using alternative port: $use_port"
        fi
    fi
    
    export FLASK_RUN_PORT=$use_port
    export FLASK_API_PORT=$use_port
    
    # Start Flask in background
    $PYTHON_CMD src/app.py > logs/flask.log 2>&1 &
    FLASK_PID=$!
    
    # Create logs directory if needed (do this BEFORE starting Flask)
    mkdir -p logs
    
    print_info "Flask API started with PID: $FLASK_PID"
    print_info "Logs: logs/flask.log"
    
    # Wait for Flask to be ready
    if wait_for_service "http://localhost:$use_port/health" "Flask API" "logs/flask.log"; then
        print_success "Flask API is running at http://localhost:$use_port"
        # Update global FLASK_PORT for Gradio to use
        FLASK_PORT=$use_port
        return 0
    else
        print_error "Failed to start Flask API"
        
        # If auto-install happened, retry once
        if [ "$AUTO_INSTALLED" = "1" ]; then
            print_info "Retrying Flask startup after dependency installation..."
            AUTO_INSTALLED=0
            
            # Clear the log file for a fresh start
            > logs/flask.log
            
            # Start Flask again
            $PYTHON_CMD src/app.py > logs/flask.log 2>&1 &
            FLASK_PID=$!
            
            print_info "Flask API restarted with PID: $FLASK_PID"
            
            if wait_for_service "http://localhost:$use_port/health" "Flask API" "logs/flask.log"; then
                print_success "Flask API is running at http://localhost:$use_port"
                FLASK_PORT=$use_port
                return 0
            else
                print_error "Failed to start Flask API on retry"
                return 1
            fi
        fi
        
        return 1
    fi
}

# Function to start Gradio
start_gradio() {
    local use_port=$GRADIO_PORT
    
    print_info "Starting Gradio interface on port $use_port..."
    
    # Check if Flask is running
    if ! check_port $FLASK_PORT; then
        print_error "Flask API is not running on port $FLASK_PORT"
        print_info "Please start Flask first: ./run.sh flask"
        return 1
    fi
    
    # Check if port is already in use
    if check_port $use_port; then
        print_warning "Port $use_port is in use by another process"
        if ! kill_port $use_port; then
            print_info "Trying to find an available port..."
            local new_port=$(find_available_port $GRADIO_ALT_PORT)
            if [ -z "$new_port" ]; then
                print_error "Cannot find an available port"
                return 1
            fi
            use_port=$new_port
            print_success "Using alternative port: $use_port"
        fi
    fi
    
    export GRADIO_SERVER_PORT=$use_port
    
    # Start Gradio in background
    $PYTHON_CMD gradio_app.py > logs/gradio.log 2>&1 &
    GRADIO_PID=$!
    
    print_info "Gradio started with PID: $GRADIO_PID"
    print_info "Logs: logs/gradio.log"
    
    # Wait a moment for Gradio to start
    sleep 3
    
    # Check if process is still running and show logs if not
    if ! ps -p $GRADIO_PID > /dev/null 2>&1; then
        print_error "Gradio process exited immediately after starting"
        show_log_on_failure "logs/gradio.log" "Gradio"
        return 1
    fi
    
    if ps -p $GRADIO_PID > /dev/null 2>&1; then
        print_success "Gradio is running at http://localhost:$use_port"
        # Update global GRADIO_PORT for display
        GRADIO_PORT=$use_port
        return 0
    else
        print_error "Gradio failed to start"
        return 1
    fi
}

# Function to run both services
run_both() {
    check_environment
    check_model
    
    print_info "Starting both Flask API and Gradio interface..."
    echo "========================================"
    
    # Start Flask
    if ! start_flask; then
        print_error "Failed to start Flask API"
        exit 1
    fi
    
    echo ""
    
    # Start Gradio
    if ! start_gradio; then
        print_error "Failed to start Gradio"
        cleanup
        exit 1
    fi
    
    echo ""
    echo "========================================"
    print_success "Both services are running!"
    echo ""
    echo -e "  ${GREEN}Flask API:${NC} http://localhost:$FLASK_PORT"
    echo -e "  ${GREEN}Gradio UI:${NC} http://localhost:$GRADIO_PORT"
    echo ""
    echo "Press Ctrl+C to stop all services"
    echo "========================================"
    
    # Wait for user interrupt
    while true; do
        sleep 1
    done
}

# Main script logic
main() {
    # Get command from arguments
    COMMAND="${1:-both}"
    
    # Handle help
    if [[ "$COMMAND" == "-h" || "$COMMAND" == "--help" ]]; then
        show_usage
        exit 0
    fi
    
    # Create necessary directories
    mkdir -p logs data models
    
    # Determine Python command
    PYTHON_CMD=$(command -v python3 || command -v python)
    
    case "$COMMAND" in
        flask|api)
            check_environment
            check_model
            start_flask
            print_success "Flask API is running. Press Ctrl+C to stop."
            while true; do sleep 1; done
            ;;
        
        gradio|ui)
            check_environment
            start_gradio
            print_success "Gradio is running. Press Ctrl+C to stop."
            while true; do sleep 1; done
            ;;
        
        both|all)
            run_both
            ;;
        
        *)
            print_error "Unknown command: $COMMAND"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
