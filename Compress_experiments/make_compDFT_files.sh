#!/bin/bash


# Specify the directory path
directory="${HOME}/Projects/NFT/compDFT"

# Check if the directory exists
if [ ! -d "$directory" ]; then
    # Create the directory
    mkdir -p "$directory"
    echo "Directory created: $directory"
else
    echo "Directory already exists: $directory"
fi


# Specify the zip file for configuration files
tar_file="./neurips_compDFT.tar"
gz_file="${tar_file}.gz"

# Check if the zip file exists
if [ -f "$gz_file" ]; then
    # Create the destination directory if it doesn't exist
    gunzip "$gz_file"

    mkdir /tmp/temp_DFT_extract
    # extract the tar file to a temporary directory
    tar xvf "$tar_file" -C /tmp/temp_DFT_extract 
    # Move the extracted files to the destination directory
    mv /tmp/temp_DFT_extract/* "$directory"
    # Remove the temporary extraction directory
    rm -r /tmp/temp_DFT_extract

    echo "Files extracted and moved to: $directory"
else
    echo "Zip file not found: $gz_file"
fi

cd $directory

