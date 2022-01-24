from ast import literal_eval
import pandas as pd
import numpy as np
from matplotlib import pyplot

tags = {
    'O': 0,
    'geo': 1,
    'gpe': 2,
    'per': 3,
    'org': 4,
    'tim': 5,
    'art': 6,
    'nat': 7,
    'eve': 8,
}
idtotags = {
    0: 'O',
    1: '#GEOGRAPHYCAL NAME#',
    2: '#GEOPOLITICAL ENTITY#',
    3: '#PERSON#',
    4: '#ORGANIZATION#',
    5: '#TIME INDICATOR#',
    6: '#ARTFACT#',
    7: '#NATURAL PHENOMENON#',
    8: '#EVENT#',
}


def preprocessY(Y, maxlen):
    Y_ready = []

    for sen_tags in Y:
        Y_ready.append(literal_eval(sen_tags))
    Y_preprocessed = []
    for strlist in Y_ready:
        new_list = [tags[item] for item in strlist]
        # find the length of the new preprocessed tag list
        len_new_tag_list = len(new_list)
        # find the differance in length between the len of tag list and padded sentences
        num_O_to_add = maxlen - len_new_tag_list
        padded_tags = new_list[:maxlen] + ([tags['O']] * num_O_to_add)
        Y_preprocessed.append(padded_tags)
    Y_preprocessed = pd.DataFrame(Y_preprocessed)
    return Y_preprocessed


def analyze_data(data):
    data = [len(tokens.split()) for tokens in data]
    data = np.array(data)
    print("Sunt in datele folosite un numar de: " + str(len(data)) + ' inputuri')
    print("Media de cuvinte pe input: " + str(np.mean(data)))
    print("Numarul maxim de cuvinte dintr-un input: " + str(np.max(data)))

    input_tokens = np.mean(data) + 2*np.std(data)
    input_tokens = int(input_tokens)
    print("Am ales numarul: " + str(input_tokens) + " ca size-ul inputului pentru reteaua noastra")

    procent_acoperire_multime = np.sum(data < input_tokens) / len(data)
    print("Acesta acopera: " + str(procent_acoperire_multime * 100) + "% din multimea de test")
    return input_tokens


def plot_history(history):
    # plot Loss
    fig, ax = pyplot.subplots(2, 1, constrained_layout=True)
    ax[0].title.set_text('Loss')
    ax[0].plot(history.history['loss'], label='train')
    ax[0].plot(history.history['val_loss'], label='validation')
    ax[0].legend()
    # plot Accuracy
    ax[1].title.set_text('Accuracy')
    ax[1].plot(history.history['accuracy'], label='train')
    ax[1].plot(history.history['val_accuracy'], label='validation')
    ax[1].legend()
    pyplot.show()


def add_NAT(predicted_input, text_input, maxlen):
    output = np.empty(shape=0, dtype=np.str)
    for sentence_index in range(0, len(text_input)):
        aux_sentence = ''
        all_words = text_input[sentence_index].split()
        covered_words = all_words[:maxlen]
        last_tag_id = -1
        for word_index in range(0, len(covered_words)):
            if word_index > 0:
                aux_sentence += ' '
            tag_id = np.argmax(predicted_input[sentence_index][word_index])
            if tag_id != 0:
                if tag_id != last_tag_id:
                    aux_sentence += idtotags[tag_id]
                else:
                    aux_sentence = aux_sentence[:-1]
                    pass
            else:
                aux_sentence += all_words[word_index]
            last_tag_id = tag_id
        for word_index in range(len(covered_words), len(all_words)):
            aux_sentence += all_words[word_index]
        output = np.append(output, aux_sentence)
    return output
