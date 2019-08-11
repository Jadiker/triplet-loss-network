'''Picks triplets for training from a bucketed dataset'''

import random
from typing import List, cast
from model_types import Example, Triplet, ExampleBuckets, Triplets

def pick_triplet(buckets: ExampleBuckets) -> Triplet:
    '''Picks a triplet for training'''
    # UPGRADE this is highly inefficient
    anchor_bucket_index = random.choice(range(len(buckets)))
    anchor_bucket = buckets[anchor_bucket_index]
    anchor = random.choice(anchor_bucket)
    # there's a small chance they're exactly the same
    positive = random.choice(anchor_bucket)
    negative_bucket_index = random.choice([bucket_index for bucket_index in range(len(buckets))
                                           if bucket_index != anchor_bucket_index])
    negative_bucket = buckets[negative_bucket_index]
    negative = random.choice(negative_bucket)

    return [anchor, positive, negative]

def pick_triplets(buckets: ExampleBuckets, how_many: int = 1) -> Triplets:
    '''Converts example buckets into triplet buckets'''
    anchors_positives_negatives: List[List[Example]] = [[], [], []]
    for triplet_index in range(how_many): # pylint: disable = unused-variable
        triplet = pick_triplet(buckets)
        for example_type_index, example_type_bucket in enumerate(anchors_positives_negatives):
            example_type_bucket.append(triplet[example_type_index])
    # cast is needed, see https://stackoverflow.com/q/56959628
    return cast(Triplets, anchors_positives_negatives)
