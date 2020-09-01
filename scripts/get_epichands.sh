#!/bin/bash

# cd scratch place
cd scratch/

# Download zip dataset from Google Drive
filename='epichands.zip'
fileid='1IrjZqY8-XnQhGIeCUq6YZBfIXjWikkZm'
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
rm ./cookie

# Unzip
unzip -q ${filename}
rm ${filename}

# cd out
cd