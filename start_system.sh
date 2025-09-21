
#!/bin/bash

# Set project directory and virtual environment path
PROJECT_DIR="/Users/atamalik/projects/agents/CarbonTrader"
VENV_PATH="/Users/atamalik/projects/agents/.venv"

# Activate virtual environment command
ACTIVATE_VENV="source $VENV_PATH/bin/activate"

# Start simulation in Terminal 1
osascript -e "tell application \"Terminal\" to do script \"$ACTIVATE_VENV && cd $PROJECT_DIR && python simulate_portfolio_update.py\""

# Start Streamlit in Terminal 2
sleep 2
osascript -e "tell application \"Terminal\" to do script \"$ACTIVATE_VENV && cd $PROJECT_DIR && streamlit run app_ui.py > frontend.log 2>&1\""

# Start initial trades in Terminal 3
sleep 2
osascript -e "tell application \"Terminal\" to do script \"$ACTIVATE_VENV && cd $PROJECT_DIR && python app_mcp.py\""

echo "ACCMN system started in new terminals. Check http://localhost:8501 for the UI."
echo "Logs: simulation.log for portfolio updates, frontend.log for Streamlit."
