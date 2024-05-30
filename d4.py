import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding

# Convert each row to a single string
input_texts = df.apply(lambda row: ' '.join(map(str, row)), axis=1).tolist()

# Create the target string as the column names
target_text = column_names_string

# Tokenize the input texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(input_texts + [target_text])
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequence = tokenizer.texts_to_sequences([target_text])[0]

# Pad sequences
max_input_length = max(len(seq) for seq in input_sequences)
max_target_length = len(target_sequence)
input_sequences = pad_sequences(input_sequences, maxlen=max_input_length, padding='post')

# Prepare target data
target_sequences = np.tile(target_sequence, (len(input_sequences), 1))

# Define model parameters
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 50
latent_dim = 256

# Define the model
encoder_inputs = Input(shape=(None,))
x = Embedding(vocab_size, embedding_dim)(encoder_inputs)
x, state_h, state_c = LSTM(latent_dim, return_state=True)(x)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
x = Embedding(vocab_size, embedding_dim)(decoder_inputs)
x = LSTM(latent_dim, return_sequences=True, return_state=False)(x, initial_state=encoder_states)
decoder_outputs = Dense(vocab_size, activation='softmax')(x)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

# Prepare the target data for training
target_data = np.expand_dims(target_sequences, -1)

# Train the model
history = model.fit([input_sequences, target_sequences], target_data, batch_size=32, epochs=10, validation_split=0.2)

# Define inference models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_inputs_single = Input(shape=(1,))
x = Embedding(vocab_size, embedding_dim)(decoder_inputs_single)
x = LSTM(latent_dim, return_sequences=True, return_state=True)(x, initial_state=decoder_states_inputs)
decoder_outputs, state_h, state_c = x
decoder_states = [state_h, state_c]
decoder_outputs = Dense(vocab_size, activation='softmax')(decoder_outputs)
decoder_model = Model([decoder_inputs_single] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Decode sequence function
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_sequence[0]

    decoded_sentence = ''
    stop_condition = False
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = tokenizer.index_word.get(sampled_token_index, '')

        decoded_sentence += ' ' + sampled_token

        if sampled_token == '' or len(decoded_sentence.split()) > max_target_length:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence.strip()

# Evaluate model
input_seq = input_sequences[0:1]  # Example: predict for the first input sequence
decoded_sentence = decode_sequence(input_seq)
print('Decoded sentence:', decoded_sentence)

# Calculate accuracy over the entire dataset
correct_predictions = 0
for i in range(len(input_sequences)):
    input_seq = input_sequences[i:i+1]
    decoded_sentence = decode_sequence(input_seq)
    if decoded_sentence == target_text:
        correct_predictions += 1

accuracy = correct_predictions / len(input_sequences)
print('Accuracy:', accuracy)