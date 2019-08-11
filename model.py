'''Creates losses and models'''

# pylint: disable = import-error, invalid-name, line-too-long
from typing import Tuple
import keras.backend as K
from keras.layers import Dense, Input, concatenate
from keras.models import Sequential, Model
from model_types import (LossFunction, Example, Examples, ExampleLength,
                         ExampleShape, EncodingLength, EncodedExamples,
                         LossInputType, LossReturnType, EncoderModel, TripletModel,
                         AccuracyReturnType, AccuracyFunction)

def get_triplet_loss(alpha: float) -> LossFunction:
    '''
    Returns the triplet loss function, given an alpha.
    Needs the encoding length in order to reconstruct each example in a concatenated triplet.
    '''
    # from https://stackoverflow.com/questions/41075993/facenet-triplet-loss-with-keras
    def triplet_loss(y_true: LossInputType, y_pred: LossInputType) -> LossReturnType: # pylint: disable = unused-argument
        '''Keras wrapper around tensorflow triplet loss'''
        # note that while y_true is not used, Keras with throw an error if it doesn't have the expected shape
        # thus, the type is still LossInputType; you still need to pass it something

        # keep batch size, but turn it into triplets
        embeddings = K.reshape(y_pred, (K.shape(y_pred)[0], 3, -1))

        positive_distance = K.mean(K.square(embeddings[:, 0] - embeddings[:, 1]), axis=-1)
        negative_distance = K.mean(K.square(embeddings[:, 0] - embeddings[:, 2]), axis=-1)
        return K.mean(K.maximum(0.0, positive_distance - negative_distance + alpha))

    return triplet_loss

def get_triplet_accuracy(alpha: float) -> AccuracyFunction:
    '''Returns a triplet accuracy function, given an alpha'''

    def triplet_accuracy(y_true: LossInputType, y_pred: LossInputType) -> AccuracyReturnType: # pylint: disable = unused-argument
        # note that while y_true is not used, Keras with throw an error if it doesn't have the expected shape
        # thus, the type is still LossInputType; you still need to pass it something
        embeddings = K.reshape(y_pred, (K.shape(y_pred)[0], 3, -1))

        positive_distance = K.mean(K.square(embeddings[:, 0] - embeddings[:, 1]), axis=-1)
        negative_distance = K.mean(K.square(embeddings[:, 0] - embeddings[:, 2]), axis=-1)

        # following the formula from the paper
        # https://arxiv.org/pdf/1503.03832.pdf
        bool_successes = K.less(positive_distance + K.ones_like(positive_distance) * alpha, negative_distance)
        return bool_successes

    return triplet_accuracy

def create_models(input_length: ExampleLength, output_length: EncodingLength) -> Tuple[TripletModel, EncoderModel]:
    '''
    Returns two models, an encoder model for prediction, and a triplet model for training.
    The triplet model uses the encoder model and trains its weights.
    '''
    input_shape: ExampleShape = (input_length,)

    anchor_input = Input(shape=input_shape)
    positive_input = Input(shape=input_shape)
    negative_input = Input(shape=input_shape)

    shared_encoder = Sequential()
    shared_encoder.add(Dense(output_length * 2))
    shared_encoder.add(Dense(output_length, activation='sigmoid'))

    anchor_encoded = shared_encoder(anchor_input)
    positive_encoded = shared_encoder(positive_input)
    negative_encoded = shared_encoder(negative_input)

    merged_outputs = concatenate([anchor_encoded, positive_encoded, negative_encoded], axis=-1)

    triplet_model: TripletModel = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_outputs)
    encoder_model: EncoderModel = Model(inputs=anchor_input, outputs=anchor_encoded)

    return triplet_model, encoder_model

def encode_example(example: Example, model: EncoderModel) -> EncodedExamples:
    return model.predict([[example]])

def encode_examples(examples: Examples, model: EncoderModel) -> EncodedExamples:
    return model.predict([examples])
