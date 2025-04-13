#!/bin/bash

# Set the output name for the executable
OUTPUT="dist/rnn"

# Define your source files
SOURCES="src/main.c src/matrix/matrix.c src/vocabulary/vocabulary.c src/model/rnn.c"

# Define any compiler flags if needed (e.g., for debugging)
CFLAGS="-Wall -g"

# Compile the source files into an executable
gcc $CFLAGS $SOURCES -o $OUTPUT -lm

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful!"

    # Run the program 
    ./$OUTPUT
else
    echo "Compilation failed."
fi
