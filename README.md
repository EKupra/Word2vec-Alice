# Word2Vec in Pure NumPy (CBOW with Negative Sampling)
This project implements the core training loop of **Word2Vec** using only **NumPy**.  

**Model variant:** CBOW (Continuous Bag of Words) with Negative Sampling  
**Dataset:** Alice’s Adventures in Wonderland by Lewis Carroll (Project Gutenberg, public domain)

## What This Implements
The implementation follows the **CBOW architecture**.

For each training example, the model:
1. Collects surrounding words within a fixed window around a target word.
2. Computes the **average embedding of the context words**.
3. Uses this averaged context representation to **predict the center word**.

## Forward Pass
Let `v` be the averaged context vector and `u` the output embedding.
**Score**
score = dot(v, u)
**Probability**
probability = sigmoid(score)

## Loss Function
The model is trained using **Negative Sampling**.
**Positive pair (target = 1)**
-log(sigmoid(score_pos))
**Negative samples (target = 0)**
-log(1 - sigmoid(score_neg))

## Gradient
The implementation uses the identity:
dL/dscore = probability − target

Positive sample:
error = prob_pos − 1

Negative sample:
error = prob_neg − 0

## Parameter Updates
Parameters are updated using **Stochastic Gradient Descent (SGD)**:
parameter -= learning_rate * gradient

Two embedding matrices are trained:
- `W_in` — input embeddings (context words)
- `W_out` — output embeddings (target words)

During CBOW training, gradients are distributed back to **all context word embeddings**, since the context representation is the average of multiple vectors.

## Negative Sampling
Negative words are sampled according to: P(word) ∝ count(word)^0.75 - This reduces computational cost compared to computing a full softmax over the entire vocabulary.

## Preprocessing
Text preprocessing includes:

- Removing Project Gutenberg header and footer
- Converting text to lowercase
- Removing non-alphabetic characters
- Removing Roman numerals used in chapter headings
- Removing rare words (minimum frequency = 5)
  
Training examples are generated using a context window around each target word.

## Hyperparameters

- Embedding dimension: **100**
- Window size: **3**
- Negative samples per positive example: **10**
- Learning rate: **0.005**
- Epochs: **5**
- Minimum word frequency: **5**

Random seeds are fixed for reproducibility.

## Evaluation

After training, word embeddings are inspected using cosine similarity to find the nearest neighbors of selected words.
Common stopwords and very frequent corpus words are filtered to make semantic similarities easier to interpret.

## How to Run

1. Download *Alice’s Adventures in Wonderland by Lewis Carroll* (UTF-8) from Project Gutenberg: https://www.gutenberg.org/ebooks/11
2. Save the file as: Alice.txt in the same directory as the script.
3. Install dependencies: pip install -r requirements.txt
4. Run the training script: python Word2vec-Alice.py
