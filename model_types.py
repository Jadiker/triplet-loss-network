'''Defines useful types for typing'''

from typing import Any, List, Tuple, Callable, Sequence
from keras.models import Model

# pylint: disable=invalid-name

Shape = Tuple[int, ...]
# numpy ndarray or Tensorflow tensor; whatever is used as input vectors for Keras
Vector = Any
# the length of a vector
# Vector.shape == (VectorLength,)
VectorLength = int
Accuracy = float

# vector with the length of one example
Example = Vector
# length of one example
ExampleLength = VectorLength
# each example should just be a flattened vector; it should have a one-dimensional shape
# since we're just encoding, not recovering, shape does not matter
ExampleShape = Tuple[ExampleLength]

Encoding = Vector
EncodingLength = VectorLength
Encodings = Sequence[Encoding]

Label = int

# len(Triplet) == 3
# [anchor, positive, negative]
Triplet = List[Example]

# the three Examples in a triplet all smashed into one vector
ConcatenatedTriplet = Vector

Labels = Sequence[Label]
Examples = List[Example]

# something that can be passed to the EncoderModel
# len(Encodable) == 1
# len(Encodable[0]) == number of examples to encode
Encodable = List[List[Example]]

# shape == (number of examples given to encode at once, EncodingLength)
# can also think of this as Sequence[Encoding]
EncodedExamples = Vector

# for example, (x_train, y_train)
Dataset = Tuple[Examples, Labels]

# [[examples labeled 0], [examples labeled 1], ...]
ExampleBuckets = List[Examples]

# len(Triplets) = 3
# [[anchors], [positives], [negatives]] such that [[anchors][i], [positives][i], [negatives][i]] forms a Triplet
Triplets = List[Examples]

# type that the loss function expects for y_true and y_pred
# LossInputType.shape = (number_of_triplets, ConcatenatedTriplet)
LossInputType = Vector
# whatever it is that loss functions return
LossReturnType = Any
# something like a float - likely a Keras tensor?
AccuracyReturnType = Any
LossFunction = Callable[[LossInputType, LossInputType], LossReturnType]
AccuracyFunction = Callable[[LossInputType, LossInputType], AccuracyReturnType]

# expects an Encodable as input for prediction, and outputs EncodedExamples on prediction
# uncompiled, and should not need to be compiled due to https://github.com/keras-team/keras/issues/3074
EncoderModel = Model

# expects Triplets as x input for training, LossInputType as y input for training,
#...and outputs LossInputType during training and for prediction.
# is not compiled yet
TripletModel = Model
