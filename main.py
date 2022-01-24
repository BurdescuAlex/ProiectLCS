import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import utility_functions as uf

if __name__ == '__main__':
    data = pd.read_csv('ner.csv')
    X = list(data['Sentence'])
    Y = list(data['Tag'])
    maxlen = uf.analyze_data(X)

    # consider the top 36000 words in the dataset
    max_words = 36000

    # tokenize each sentence in the dataset
    tokenizer = Tokenizer(num_words=max_words, filters='')
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    word_index = tokenizer.word_index
    print("Found {} unique tokens.".format(len(word_index)))
    ind2word = dict([(value, key) for (key, value) in word_index.items()])

    word2id = word_index
    # dict. that map each identifier to its word
    id2word = {}
    for key, value in word2id.items():
        id2word[value] = key

    X_preprocessed = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
    Y_preprocessed = uf.preprocessY(Y, maxlen)

    # 70% of the data will be used for training
    training_samples = 1
    # 15% of the data will be used for validation
    validation_samples = 0
    # 15% of the data will be used for testing
    testing_samples = 0

    X_preprocessed = np.asarray(X_preprocessed)
    Y_preprocessed = np.asarray(Y_preprocessed)

    indices = np.arange(len(Y_preprocessed))
    np.random.seed(seed=123)
    np.random.shuffle(indices)

    X_preprocessed = X_preprocessed[indices]
    Y_preprocessed = Y_preprocessed[indices]
    X_train = X_preprocessed[: int(training_samples * len(X_preprocessed))]
    print("Number of training examples: {}".format(len(X_train)))

    X_val = X_preprocessed[int(training_samples * len(X_preprocessed)): int(training_samples * len(X_preprocessed)) + (
            int(validation_samples * len(X_preprocessed)) + 1)]
    print("Number of validation examples: {}".format(len(X_val)))

    X_test = X_preprocessed[int(training_samples * len(X_preprocessed)) + (int(testing_samples * len(X_preprocessed)) + 1):]
    print("Number of testing examples: {}".format(len(X_test)))

    Y_train = Y_preprocessed[: int(training_samples * len(X_preprocessed))]
    Y_val = Y_preprocessed[int(training_samples * len(X_preprocessed)): int(training_samples * len(X_preprocessed)) + (
            int(validation_samples * len(X_preprocessed)) + 1)]
    Y_test = Y_preprocessed[int(training_samples * len(X_preprocessed)) + (int(testing_samples * len(X_preprocessed)) + 1):]

    print("Total number of examples after shuffling and splitting: {}".format(len(X_train) + len(X_val) + len(X_test)))

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

    BATCH_SIZE = 132
    SHUFFLE_BUFFER_SIZE = 132

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    val_dataset = val_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    embedding_dim = 32
    max_words = len(word_index) + 1

    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(max_words, embedding_dim, input_length=maxlen),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, activation='relu', return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=32, activation='relu', return_sequences=True)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(9, activation='softmax'))
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    # history = model.fit(train_dataset,
    #                     validation_data=val_dataset,
    #                     epochs=7)
    # model.evaluate(test_dataset)
    model = tf.keras.models.load_model("model.h5")
    # model.save("model.h5")

    text_input = np.loadtxt('input.txt', dtype=np.str, delimiter='\n', encoding='utf-8')
    tokenized_input = tokenizer.texts_to_sequences(text_input)
    padded_input = pad_sequences(tokenized_input, maxlen=maxlen, padding='post', truncating='post')
    predicted_input = model.predict(padded_input)
    output = uf.add_NAT(predicted_input, text_input, maxlen)
    # print('\n'.join(text_input))
    print('\n'.join(output))  # nit edit to make everything look cleaner - Vlad
    # uf.plot_history(history)
