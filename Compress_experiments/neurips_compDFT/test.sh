#!/bin/bash


# Specify the directory path
directory="${HOME}/Projects/NFT"

# Check if the directory exists
if [ ! -d "$directory" ]; then
    # Create the directory
    mkdir -p "$directory"
    echo "Directory created: $directory"
else
    echo "Directory already exists: $directory"
fi
