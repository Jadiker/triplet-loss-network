'''Functions for cleaning data'''

from typing import List
from model_types import (Example, ExampleBuckets, Dataset, EncodingLength, Triplet,
                         Triplets, LossInputType)

def dataset_to_buckets(dataset: Dataset) -> ExampleBuckets:
    '''
    Turns a dataset of (examples, labels) into a list of buckets:
    [[examples labeled 0], [examples labeled 1], ...]
    '''
    examples, labels = dataset
    example_buckets: List[List[Example]] = []
    max_label = max(labels)
    for bucket_index in range(max_label + 1): # pylint: disable = unused-variable
        example_buckets.append([])

    for example_index, example_value in enumerate(examples):
        bucket_number = labels[example_index]
        value = example_value.flatten() # assumes that its a numpy ndarray
        example_buckets[bucket_number].append(value)

    assert True not in [len(bucket) < 2 for bucket in example_buckets], "Each label must have at least two examples to be trained with triplet loss"

    return example_buckets

def get_zeroed_pred_triplets(amount: int, output_size: EncodingLength) -> LossInputType:
    '''
    Returns a list of zeroed triplets that fools Keras into thinking this is the true predicted values
    Generally used because using K.zeros_like causes my computer to freeze for some reason.
    '''

    return [[[0] * output_size * 3] * amount]

def isolate_triplet(triplet_index: int, triplets: Triplets) -> Triplets:
    '''
    Isolates a particular triplet from within many triplets
    Return value can be passed to the triplet model
    '''
    return [[triplets[0][triplet_index]], [triplets[1][triplet_index]], [triplets[2][triplet_index]]]

def get_triplet(triplet_index: int, triplets: Triplets) -> Triplet:
    '''
    Gets a particular triplet from triplets
    Return value can be passed to the encoder model
    '''
    return [triplets[0][triplet_index], triplets[1][triplet_index], triplets[2][triplet_index]]
