import numpy as np
import os
import time

import tensorflow as tf


class MyModel(tf.keras.Model):

  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    """
    Call the model for training.
    """
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)
    return (x, states) if return_state else x


class OneStep(tf.keras.Model):


  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    """
    Generate one step training.
    """
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states


with open('src/PoliticianBeliefDoc.txt', 'r') as f:
    text = f.read()

# length of text is the number of characters in it
print(f'Length of text: {len(text)} characters')
# Take a look at the first 250 characters in text
# print(text[:2500]) # We can change depending on how many words we have in the text file.
# The unique characters in the file
vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')

example_texts = ['abcdefg', 'xyz']
chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
chars
# Converts from tokens to character IDs:
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)
ids = ids_from_chars(chars)

# Inverts this representation and recover human-readable strings from it
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
    #recovers the characters from the vectors of IDs, and returns them as a tf.RaggedTensor of characters
chars = chars_from_ids(ids)
# Uses tf.strings to put the characters back into a string
tf.strings.reduce_join(chars, axis=-1).numpy()
def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

# Prediction task

# convert the text vector into a stream of character indices.
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
for ids in ids_dataset.take(10):
    print(chars_from_ids(ids).numpy().decode('utf-8'))

#  convert these individual characters to sequences of the desired size
seq_length = 100
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
for seq in sequences.take(1):
  print(chars_from_ids(seq))
# joins the tokens back into strings
for seq in sequences.take(5):
  print(text_from_ids(seq).numpy())

# takes a sequence as input, duplicates, and shifts it to align the input and label for each timestep
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

split_input_target(list('Tensorflow')) # Probably change or delete later
dataset = sequences.map(split_input_target)
for input_example, target_example in dataset.take(1):
    print('Input :', text_from_ids(input_example).numpy())
    print('Target:', text_from_ids(target_example).numpy())

# Training batches

# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))


# Build models (Will probably need to adjust values)

# Length of the vocabulary in StringLookup Layer
vocab_size = len(ids_from_chars.get_vocabulary())

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

# Define model
model = MyModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

# Testing the model (Can probably delete later)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, '# (batch_size, sequence_length, vocab_size)')

# Prints the model summary
# model.summary()

# sample from the output distribution, to get actual character indices. 
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
# This gives us, at each timestep, a prediction of the next character index:
sampled_indices

# Decode these to see the text predicted by this untrained model:
print('Input:\n', text_from_ids(input_example_batch[0]).numpy())
print()
print('Next Char Predictions:\n', text_from_ids(sampled_indices).numpy())

# Train the model

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
print('Prediction shape: ', example_batch_predictions.shape, ' # (batch_size, sequence_length, vocab_size)')
print('Mean loss:        ', example_batch_mean_loss)

# check that the exponential of the mean loss is approximately equal to the vocabulary size. A much higher loss means the model is sure of its wrong answers, and is badly initialized:
tf.exp(example_batch_mean_loss).numpy()

# Configure the training procedure using the tf.keras.Model.compile method.
model.compile(optimizer='adam', loss=loss)

# Configure Checkpoints

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# Excute training

# Training (Amount of times the ai should train)
EPOCHS = 50
# Shows training results
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

start = time.time()
states = None
next_char = tf.constant(['My fellow Americans'])
result = [next_char]

for _ in range(1000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)

# Export generator

# This single-step model can easily be saved and restored, allowing you to use it anywhere a tf.saved_model is accepted.
tf.saved_model.save(one_step_model, 'one_step')
one_step_reloaded = tf.saved_model.load('one_step')

states = None
next_char = tf.constant(['My fellow Americans'])
result = [next_char]

for _ in range(200):
  next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
  result.append(next_char)

print(tf.strings.join(result)[0].numpy().decode('utf-8'))
