'''Main program to run and test'''

import os
import random
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.datasets import mnist
from keras.optimizers import Adam
import model as md
from triplet_mining import pick_triplets
from display import display_images, display_training_history
from data_cleaning import dataset_to_buckets, get_triplet, get_zeroed_pred_triplets
from np_triplet_loss import triplet_loss as np_triplet_loss
from model_types import EncodingLength, ExampleLength

# Setting seed value
# from https://stackoverflow.com/a/52897216
SEED = 42
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(SEED)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(SEED)
# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(SEED)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.set_random_seed(SEED)
# 5. Configure a new global `tensorflow` session
# if you want to run the code in parallel, you may want to change these settings
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# set how channels are done - useful for RGB images
K.set_image_data_format('channels_last')

# constants
TRAIN_ALPHA: float = .6
TEST_ALPHA: float = 0
# how big the output vector is
OUTPUT_LENGTH: EncodingLength = 100
TRAIN_TRIPLET_AMT = 100000
TEST_TRIPLET_AMT = 10000
LEARNING_RATE = .0001
EPOCHS = 40
BATCH_SIZE = 32

(x_train, y_train), (x_test, y_test) = mnist.load_data()

max_label = max(y_train)
assert max_label == 9, "max label was not the expected 9"

train_buckets = dataset_to_buckets((x_train, y_train))
test_buckets = dataset_to_buckets((x_test, y_test))
train_triplets = pick_triplets(train_buckets, TRAIN_TRIPLET_AMT)
test_triplets = pick_triplets(test_buckets, TEST_TRIPLET_AMT)

first_triplet = get_triplet(0, train_triplets)
display_images(first_triplet)

input_shape = get_triplet(0, train_triplets)[0].shape
assert len(input_shape) == 1, f"Shape {input_shape} should have only had one dimension - it should have already been flattened"
input_length: ExampleLength = input_shape[0]

# create the models - one for training, one for prediction; uses shared layers
training_model, prediction_model = md.create_models(input_length=input_length, output_length=OUTPUT_LENGTH)

encoded_triplet = md.encode_examples(first_triplet, prediction_model)
anchor, positive, negative = encoded_triplet
print(f"Positive loss before training: {np_triplet_loss(anchor, positive)}")
print(f"Negative loss before training: {np_triplet_loss(anchor, negative)}")

display_images(encoded_triplet)

# create the loss function
triplet_loss = md.get_triplet_loss(TRAIN_ALPHA)
print("Compiling....")

# compile the model for training
training_model.compile(loss=triplet_loss, optimizer=Adam(lr=LEARNING_RATE),
                       metrics=[md.get_triplet_accuracy(TEST_ALPHA)]
                       )
print("Training...")
# train the model - the will train the prediction model as well
history = training_model.fit(x=train_triplets, y=get_zeroed_pred_triplets(TRAIN_TRIPLET_AMT, OUTPUT_LENGTH),
                             validation_data=(test_triplets, get_zeroed_pred_triplets(TEST_TRIPLET_AMT, OUTPUT_LENGTH)),
                             epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

anchor, positive, negative = md.encode_examples(first_triplet, prediction_model)

print(f"Positive loss after training: {np_triplet_loss(anchor, positive)}")
print(f"Negative loss after training: {np_triplet_loss(anchor, negative)}")

display_images([anchor, positive, negative])

display_training_history(history)
