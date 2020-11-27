# In this file I have implemented text classification task with
# Multi-channel CNN architecture
# Highkights :
#       Use of Dataset object (write from step-1 ) from directory
#       Customised cleaning of each element of dataset
#       Customised Tokensization of each element of dataset
#       MultiChannel CNN Model
#
import os
import tensorflow as tf
import numpy as np
import pickle
from string import punctuation
from tensorflow.keras.preprocessing import text_dataset_from_directory
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Flatten, Conv1D, Dense, Dropout, MaxPooling1D, concatenate, Embedding
from tensorflow.keras.models import Model


def datset_objbuilder(main_dir_path, debuggining_info=True):
    """

    :param main_dir_path: Path to main directory
    :return:
    """
    complete_dataset = text_dataset_from_directory(
        main_dir_path,
        class_names=['neg', 'pos'],
    )

    if debuggining_info:
        print("Accessing and Building Dataset....")
        print("First_batch_of_dataset..")
        print(list(complete_dataset.take(1).as_numpy_iterator()))  # 32 files (batch_size)
        print(complete_dataset.element_spec)
        print("Dataset Building Successful")
        for files, targets in complete_dataset.take(1).as_numpy_iterator():
            print("First File content: ")
            print(files[0])
            print("Target: ")
            print(targets[0])
            break

    return complete_dataset


main_dir_path = "./dataset/txt_sentoken"
complete_dataset = datset_objbuilder(main_dir_path)


def clean_doc(doc):
    """
    Specifies how to clean the doc and return tokens

    :param doc: doc a single string content
    :return: 
    """""
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def word_counter_builder(clean_doc, complete_dataset, debugging_info=True):
    """
    Builds word count in the entire dataset
    :param clean_doc: Function that returns TOkens and Specifies how to clean and TOkensise the data
    :param complete_dataset: dataset_object
    :return: word_count (dictionary)
    """
    batch_counter = 0
    file_counter = 0
    word_count = dict()
    # x_dataset = complete_dataset.as_numpy_iterator()  #64 batches ka dataset_obj
    # print("----x_dataset")
    # print(len(list(x_dataset)))

    for batch_content, batch_target in complete_dataset.as_numpy_iterator():
        batch_counter += 1
        print("Batch Accessed............")
        # print("Batch_content_size: ",batch_content.shape)
        # print("Batch_Target_size: ",batch_target.shape)

        for content in batch_content:
            print("Batch_files Accessed")
            file_counter += 1
            tokens = clean_doc(str(content))
            for word in tokens:
                if word not in word_count:
                    word_count[word] = 0
                word_count[word] += 1

    if debugging_info:
        print("Total batches_accessed: ", batch_counter)
        print("Total Files Accessed: ", file_counter)
        sorted_word_count = {k: v for k, v in sorted(word_count.items(), key=lambda item: item[1], reverse=True)}
        tmp = 0
        print("First top 10 elements by occcurence")
        for element in sorted_word_count.items():
            tmp += 1
            if tmp <= 10:
                print(element)
        print("Keys in total word count: ", len(list(word_count.keys())))
    return word_count


word_count = word_counter_builder(clean_doc, complete_dataset)
print("Word_count : ", len(word_count.items()))


def word_count_slicer_Token_list_generator(word_count, num_words=20000, debugging_info=True):
    """
    :param word_count: All vocab Word count
    :param num_words: These many most occuring words will be kept from total Voccab
    :return:
    """
    sorted_word_count = {k: v for k, v in sorted(word_count.items(), key=lambda item: item[1], reverse=True)}
    tmp_counter = 0
    vocab = []
    updated_word_count = {}
    for key, value in sorted_word_count.items():
        tmp_counter += 1
        if tmp_counter <= num_words:
            vocab.append(key)
            updated_word_count[key] = value
    if debugging_info:
        print("First top 10 element by occcurence in Updated Word counter")
        tmp_1 = 0
        for element in updated_word_count.items():
            tmp_1 += 1
            if tmp_1 <= 10:
                print(element)
        print("Keys in Updated Word Count: ", len(list(updated_word_count.keys())))

    return sorted(vocab), updated_word_count


tokens = word_count_slicer_Token_list_generator(word_count)[0]
updated_word_count = word_count_slicer_Token_list_generator(word_count)[1]


def create_mappings(tokens, debuggining_info=True):
    """

    :param tokens: sorted tokens list
    :return: word_to_idx and idx_to_word
    """
    word_idx = {}
    for i, word in enumerate(tokens, start=2):  # Leave 0 for padding and 1 for unkown tokens
        word_idx[word] = i
    word_idx["<UNK>"] = 1

    idx_word = {}
    for i, word in enumerate(tokens, start=2):
        idx_word[i] = word
    idx_word[1] = "<UNK>"

    if debuggining_info:
        print("Length of Word index: ", len(word_idx.items()))  # num_words + 1 (Unknown Token / 0 padding)
        print("Length of index word mapping: ", len(idx_word.items()))  # num_words + 1 (UNK TOken/ (0--padding)
        print("Length of tokens: ", len(tokens))  # num_words
        print("length of updated word counter : ", len(updated_word_count.items()))  # num_words

    return word_idx, idx_word


word_to_idx, idx_to_word = create_mappings(tokens)

vocab_size = len(tokens) + 2  # (num_words + 1 (unknown token) + 1 (0 padding to make all equal length)

# Tokenising Content ##
print("-----------------Tokenising COntent---------------------")


def Tokenise_content(word_to_idx, complete_dataset, file_path, debugging_info=True):
    """

    :param word_to_idx: Maps word to idx
    :return: Tokenised_content
    """
    batch_counter_1 = 0
    file_counter_1 = 0
    tokenised_content = []

    for BATCH_CONTENT, BATCH_TARGET in complete_dataset.as_numpy_iterator():
        batch_counter_1 += 1
        print("Batch Accessed............")
        print(batch_counter_1)
        # print("Batch_content_size: ",batch_content.shape)
        # print("Batch_Target_size: ",batch_target.shape)
        # We can use here parallel computing to Tokenise the batches directly

        for CONTENT in BATCH_CONTENT:
            file_counter_1 += 1
            print("Files Accessed So Far")
            print(file_counter_1)
            print("Working on File")
            encoded_file_content = []
            word_counter = 0
            for word in CONTENT.split():
                word = word.decode('UTF-8')
                word_counter += 1
                if word in tokens:
                    encoded_file_content.append(word_to_idx[word])
                else:
                    encoded_file_content.append(1)
            tokenised_content.append(encoded_file_content)

            if debugging_info:
                print("-------------------------------------")
                print("Actual File Content: ")
                print(CONTENT.split())
                print("Encoded File Content: ")
                print(encoded_file_content)
                print("File Length: ", word_counter)
                print("Encoded File Length: ", len([word for word in encoded_file_content]))

    print("Saving Encoded File Content")

    file_path = file_path

    with open(file_path, "wb") as f:
        pickle.dump(tokenised_content, f)
    print("File Saved Successfully")


def file_loader_saver(file_path, task, saving_content=None):
    """
    Only for .pkl files
    :param file_path: file_path
    :param task: saving / loading
    :return:
    """
    file_path = file_path
    if task == "saving":
        if saving_content is not None:
            print("Saving Content")
            with open(file_path, 'wb') as f:
                pickle.dump(saving_content, f)

            print("File saved successfully")
            print("File Saved here: ")
            print(file_path)

    if task == 'loading':
        print("Loading File")
        with open(file_path, 'rb') as f:
            file = pickle.load(f)

        return file


file_path_tokenised_content = "./tokenised_content.pkl"

tokenised_content = file_loader_saver(file_path_tokenised_content, "loading")

# Tokenise_content(word_to_idx, complete_dataset, file_path = file_path_tokenised_content,debugging_info=False)
#
# print("Loading Tokenised File Content: ")
#
# with open(file_path_tokenised_content,'rb') as f:
#     tokenised_content = pickle.load(f)
#
# print("Loading Successful ")
# This tokenised content is of variable lengths
# This tokenised content is of shape (2000 (# of files), varibale length for each sequence)
# tokenised_content_np = np.array(tokenised_content)
# print(tokenised_content_np.shape)
print(len(tokenised_content[0]))
print(len(tokenised_content[1]))

padded_tokenised_content = pad_sequences(tokenised_content)

processed_content = np.array(padded_tokenised_content)
print(processed_content.shape)
# print(processed_content[0][])

padded_tokenised_content_file_path = "./padded_tokenised_content.pkl"
file_loader_saver(padded_tokenised_content_file_path, "saving", saving_content=processed_content)

max_doc_length = processed_content.shape[1]

print("-----------------------Working on Model------------------------------")


# define the model
def define_model(length, vocab_size):
    """

    :param length: input_length
    :param vocab_size: vocab_size
    :return:
    """
    # channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, 100)(inputs1)
    conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    # channel 2
    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size, 100)(inputs2)
    conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    # channel 3
    inputs3 = Input(shape=(length,))
    embedding3 = Embedding(vocab_size, 100)(inputs3)
    conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)
    # merge
    merged = concatenate([flat1, flat2, flat3])
    # interpretation
    dense1 = Dense(10, activation='relu')(merged)
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print(model.summary())
    # plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model


model_1 = define_model(max_doc_length, vocab_size)


# def checkpoint_callback_creater(checkpoint_dir):
#     """
#
#     :param checkpoint_dir: Directory where checkpoint will be saved
#     :return:
#     """
#     # Configuring Checkpoints
#     # Directory where the checkpoints will be saved
#     checkpoint_dir = checkpoint_dir
#     # Name of the checkpoint files
#     checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
#
#     checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#         filepath=checkpoint_prefix,
#         save_weights_only=True)
#
#     return checkpoint_callback


checkpoint_dir = "./moview_review_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# checkpoint_callback = checkpoint_callback_creater(chckpoint_dir)
targets_1  = list([element[1] for element in complete_dataset.as_numpy_iterator()])
targets = []
for element in complete_dataset.as_numpy_iterator():
    # element -- > 1 batch of Inputs , Targets
    batched_targets = element[1]
    # Since Last Batch is of 16 size it'll create during transition form list to np.array
    for target in batched_targets:
        targets.append(target)

print(len(targets))
targets_np = np.array(targets)
print(targets_np.shape)
# processed_content_ds = tf.data.Dataset.from_tensor_slices(processed_content)
# targets_ds = tf.data.Dataset.from_tensor_slices(targets_np)
# processed_dataset_gen = tf.data.Dataset.zip((processed_content_ds,targets_ds))
# # Creating New processed Dataset obj
# print("-------------------------------------------------------------------------")
# print("Processed COntent: ")
# print(processed_content[0])
# print("----------------------------------------------------------------------------")
# print("Processed Dataset 1st element :")
# print(list(processed_dataset_gen.take(1).as_numpy_iterator()))
# print("----------------------------------------------------------------------------")
# print("ACtual Dataset 1st element")
# print(list(complete_dataset.take(1).as_numpy_iterator()))

#history = model_1.fit([processed_content,processed_content,processed_content],targets_np, epochs=10, callbacks=[checkpoint_callback],validation_split=0.2)

model_1.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model_1.build(input_shape=(max_doc_length,vocab_size))

def model_evaluator(model,word_to_idx, evaluation_line="Awesome",debugging_info =True):
    """

    :param evaluate_line: Line to Evaluate
    :param model: Model to Make Predictions from
    :param word_to_idx: Mapping
    :return: Prediction
    """
    evaluate_line = evaluation_line

    # Preprocessing evaluate line
    encoded_evaluation_line = []
    for word in evaluate_line.split():
        if word in tokens:
            idx = word_to_idx[word]
            encoded_evaluation_line.append(idx)
        else:
            idx = 1  # Token for unknown word
            encoded_evaluation_line.append(idx)

    encoded_evaluation_line = np.reshape(encoded_evaluation_line, (1, len(encoded_evaluation_line)))
    processed_evaluated_line = pad_sequences(encoded_evaluation_line,maxlen=max_doc_length)

    if debugging_info:
        print("Encoded Evaluation Line; ")
        print(encoded_evaluation_line)
        print("Encoded Line Shape: ")
        print(np.array(processed_evaluated_line).shape)

    pred = model_1.predict(x = [processed_evaluated_line,processed_evaluated_line,processed_evaluated_line])
    if pred < 0.5:
        print("Predicted Negative Class ")
        print(pred)
    else:
        print("Predicted Positive Class")
        print(pred)

eval_line = " in 2176 on the planet mars police taking into custody an accused murderer face the title menace . there is a lot of fighting "
# eval line from cv0005_29357 -- > Negative Class


model_evaluator(model_1,word_to_idx=word_to_idx
                ,evaluation_line=eval_line)

