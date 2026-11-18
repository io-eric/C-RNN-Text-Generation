# RNN Text Generation Model in C

This repository contains a simple Recurrent Neural Network (RNN) implemented from scratch in C, designed to generate text based on a given input word. The model is trained on a small dataset of sentences and learns to predict the next word in a sequence. For example, given the input word "Matrix," the model might predict "dimensions" as the next word.

## Overview
This project implements a basic RNN model in C to generate text. The model is trained on a small dataset of sentences and learns to predict the next word in a sequence. The implementation includes:
- Vocabulary creation and word indexing.
- One-hot encoding for input and target vectors.
- Forward and backward propagation for training.
- Text generation based on a given input word.

## Features
- From Scratch Implementation: The RNN is implemented entirely in C without relying on external machine learning libraries.
- Text Generation: Given an input word, the model predicts the next word in the sequence.
- Customizable Training: Adjustable parameters such as learning rate, hidden layer size, and number of epochs.

## Requirements
- C Compiler: GCC or any C99-compatible compiler.
- Basic C Libraries: Standard libraries like stdio.h, stdlib.h, and string.h.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/io-eric/C-RNN-Text-Generation.git
   cd C-RNN-Text-Generation
   ```
2. Compile and run the project:
    ```
   ./build.sh
    ```
## Usage
After compiling and running the program, the model will:
1. Train on the provided dataset.
2. Generate text based on the input word "Matrix."
3. Print the generated text to the console.

To modify the input word, the number of generated words or how random the output is, edit the following lines in main.c:
````c
const char *input_text = "Matrix"; // Change the input word
double temperature = 0.8; // 0 = always pick the most likely word, higher = more random
char *next_word_predictions = rnn_generate_text(v, rnn, (char *)input_text, 10, temperature); // Change the number of words to generate
````

## Training Data
The model is trained on a small dataset of sentences:
```c
const char *training_data[] = {
    "Matrix dimensions don’t match? Shocking.",
    "Rain on the window? Wow, never seen that before.",
    "Starting is the hardest part? Groundbreaking insight.",
    "Traveling? Because who wouldn’t want to get lost in a new place?",
    "Books? Oh yeah, they’re just full of ideas or whatever.",
    "Time’s too short for pointless stuff... unless it’s procrastination."
};
```

You can replace this dataset with your own text data for custom training.

## Results
After training, the model generates text based on the input word. For example (actual numbers and predictions will vary between runs):
```c
$ ./build.sh
Compilation successful!
Epoch 0, Average Loss: 4.038023
Epoch 1000, Average Loss: 0.207947
Epoch 1999, Average Loss: 0.198872
Input text: Matrix
Next word predictions: dimensions don’t match? Shocking.
```

The loss is cross-entropy, so it starts around ln(vocabulary size) and does not go to 0: the first word of a sentence (after `<bos>`) can't be predicted from the context. Generation stops when the model predicts `<eos>`, so it can return fewer words than requested.

The model's performance is limited by the small dataset and simple architecture, but it demonstrates the basic principles of RNNs and text generation.

## Limitations
- Small Dataset: The model is trained on a very small dataset, which limits its ability to generalize.
- Basic Architecture: The RNN is a simple implementation and may struggle with long-term dependencies.
- Overfitting: Due to the small dataset, the model may overfit and repeat words.

## Future Improvements
- Larger Dataset: Train the model on a larger and more diverse dataset.
- Advanced Architectures: Implement more advanced RNN variants like LSTMs or GRUs.
- Better Text Generation: Improve the text generation logic to produce more coherent and varied outputs.
- User Interface: Add a command-line interface for easier interaction with the model.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
