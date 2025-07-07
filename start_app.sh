#!/bin/bash

echo "Starting LCSOD Tool..."
echo

# Check if virtual environment exists
if [ ! -d "LCSOD" ]; then
    echo "Creating virtual environment 'LCSOD'..."
    python3 -m venv LCSOD
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Please ensure Python3 is installed."
        read -p "Press any key to continue..."
        exit 1
    fi
    echo "Virtual environment created successfully."
else
    echo "Virtual environment 'LCSOD' already exists."
fi

echo
echo "Activating virtual environment..."
source LCSOD/bin/activate

echo
echo "Installing/updating dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo
echo "Starting Streamlit app..."
python -m streamlit run app/main.py

# Keep the terminal open if there's an error
if [ $? -ne 0 ]; then
    read -p "Press any key to continue..."
fi 