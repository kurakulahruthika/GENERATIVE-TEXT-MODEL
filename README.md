# GENERATIVE-TEXT-MODEL

*COMPANY*:CODTECH IT SOLUTIONS

*NAME*:KURAKULA HRUTHIKA

*INTERN ID*:CT04DF664

*DOMAIN*: AI

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH



## DESCRIPTION ABOUT MY TASK


## 📘 Generative Text Model – Project Description

### 🔍 Introduction

In this project, I set out to build a *Generative Text Model* that could generate human-like paragraphs based on a given topic or prompt. The goal was not just to produce random strings of words, but to construct meaningful and grammatically correct text that maintains coherence and context across sentences. This task provided a hands-on opportunity to apply deep learning in the field of *Natural Language Processing (NLP)*.

I approached the project using both *LSTM-based sequence modeling* and *GPT-based transformer fine-tuning*, to understand the strengths and working of each method.

---

###  How I Did It – Step-by-Step Process

####  Step 1: Understanding the Objective

Before writing any code, I researched how generative text models work, focusing on:

* How sequence prediction works using Recurrent Neural Networks (RNNs) and LSTMs
* How modern transformer models like GPT generate text using attention mechanisms
* The difference between character-level and word-level models
  This helped me plan my approach and choose the right tools.

####  Step 2: Data Collection and Preprocessing

I either created or selected a clean, topic-specific text dataset. The dataset needed to be large enough for the model to learn meaningful patterns. I used Python scripts to:

* Clean the text (remove punctuation, special characters, and convert to lowercase)
* Tokenize the text into sequences using Tokenizer from Keras (for LSTM)
* Create input-output pairs for training, where each input is a sequence of words, and the output is the next word in the sequence
* For GPT, I used Hugging Face’s tokenizer to encode the data for fine-tuning

####  Step 3: Building the LSTM Model

Using TensorFlow and Keras, I built a sequential model:

* *Embedding layer* to convert words into dense vectors
* *LSTM layers* to learn from the sequences
* *Dense layer with softmax activation* to predict the next word

I used:

* *Categorical cross-entropy* as the loss function
* *Adam optimizer* for gradient descent
* *ModelCheckpoint* and *EarlyStopping* callbacks to save the best version

I trained the model over multiple epochs while adjusting batch sizes and sequence lengths. I experimented with different numbers of LSTM layers and hidden units to strike a balance between underfitting and overfitting.

####  Step 4: Text Generation Logic

Once the model was trained, I created a loop to:

* Take a user-provided seed/prompt
* Predict the next word using the model
* Append that prediction to the input
* Repeat the process to generate multiple words/sentences

I also experimented with *temperature sampling* to adjust the creativity and randomness of the output.

####  Step 5: Implementing a GPT-Based Model

To explore modern transformers, I fine-tuned a *GPT-2 model* using the transformers and datasets libraries from Hugging Face:

* Loaded GPT2LMHeadModel and GPT2Tokenizer
* Created a dataset of text prompts and completions
* Trained the model with appropriate hyperparameters and trained on my custom dataset

The GPT model provided much better fluency and longer context retention in the generated text.

---

### Evaluation

I evaluated the outputs based on:

* Coherence (did the sentences make sense?)
* Relevance to the prompt
* Creativity (was the text diverse or repetitive?)
* Grammatical correctness

The LSTM model was good at local pattern prediction but struggled with long context. GPT outperformed LSTM in generating long-form coherent text.

---

###  Challenges Faced

* Managing memory usage during GPT fine-tuning
* Avoiding overfitting with smaller datasets
* Keeping output coherent while encouraging diversity
* Handling sequence padding and tokenization edge cases

---

###  Tools and Libraries Used

* Python 3
* TensorFlow & Keras
* Hugging Face Transformers (transformers, datasets)
* NumPy, Pandas, Matplotlib (for analysis and visualization)
* Jupyter Notebook for experiments and testing

---

### What I Learned

* How sequence models like LSTMs predict the next word in a sentence
* How to preprocess and tokenize data for NLP tasks
* How transformer models like GPT are trained and fine-tuned
* The importance of sampling techniques in text generation
* How temperature affects the diversity of generated outputs
* The real-world applications of generative text models (chatbots, story generation, auto content creation, etc.)

---

###  Conclusion

This project helped me understand the internal mechanics of generative text models and how to implement them in real projects. I was able to build a complete pipeline—from raw text data to a trained model capable of producing original text. I plan to continue enhancing the model by integrating attention mechanisms into the LSTM model and further exploring transformer-based architectures.

---
