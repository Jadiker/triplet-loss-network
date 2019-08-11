'''Display data'''
import os
from typing import Optional, Sequence

import numpy as np
from matplotlib import pyplot as plt
from model_types import Shape, Vector
from factoring import get_square_factors, get_middle_factors

def plt_show() -> None: # pylint: disable=too-many-branches
    '''Text-blocking version of plt.show() with an option to save the file'''
    no_responses = ("n", "no")
    yes_responses = ("y", "yes")
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)

    class BadSave(ValueError):
        '''Error class for when saving fails'''
        pass # pylint: disable = unnecessary-pass

    def save_figure(path):
        try:
            plt.savefig(path)
            print(f"Figure saved! The location is: {path}")
        except Exception as e:
            print("Saving there did not work, here's what went wrong:")
            print(f"{type(e).__name__}: {e}")
            raise BadSave from e

    while True: # pylint: disable=too-many-nested-blocks
        response = input("Would you like to save this image (y/n)?  (respond after viewing; empty response means 'n') ")
        if not response or response in no_responses:
            break
        elif response in ("y", "yes"):
            while True:
                current_directory = os.getcwd()
                print(f"The current directory is {current_directory}")
                response = input("Where would you like to save the image? (relative to the current directory; empty response to cancel) ")
                if not response:
                    break

                path = os.path.abspath(response)
                worked = True
                if os.path.lexists(path):
                    while True:
                        response = input("Something already exists there. Would you like to overwrite it (y/n)? ")
                        if response in yes_responses:
                            try:
                                save_figure(path)
                                break
                            except BadSave:
                                worked = False
                                break
                        elif response in no_responses:
                            worked = False
                            break
                        else:
                            print(f"Unrecognized response '{response}'. Please respond with either 'n' or 'y'.")
                else:
                    try:
                        save_figure(path)
                    except BadSave:
                        worked = False

                if worked:
                    # no failures above
                    break
            break
        else:
            print(f"Unrecognized response '{response}'. Please respond with either 'n' or 'y'.")

    plt.close()

def imshow(data: Vector, shape: Optional[Shape] = None) -> None:
    '''Put an image onto the plot'''
    if shape is None:
        shape = get_middle_factors(np.asarray(data).size)

    plt.imshow(np.reshape(data, shape), interpolation='nearest')

def display_image(data: Vector, shape: Optional[Shape] = None) -> None:
    '''Display an image'''
    imshow(data, shape)
    plt_show()

def display_images(data: Sequence[Vector], shapes: Optional[Sequence[Optional[Shape]]] = None) -> None:
    '''Display a sequence of images'''
    if shapes is None:
        shapes = [None] * len(data)

    data = list(data)
    number_of_plots = len(data)
    dimensions = get_square_factors(number_of_plots)
    def subplot_at(index):
        plt.subplot(dimensions[0], dimensions[1], index + 1)

    for data_index, data_value in enumerate(data):
        subplot_at(data_index)
        imshow(data_value, shapes[data_index])

    plt_show()


def split_term(s):
    '''
    Determines whether a metric is a train or test metric
    returns (("Test" or "Train"), metric name)
    '''
    if s.startswith('val_'):
        return "Test", s[len('val_'):]
    else:
        return "Train", s

def title_term(s):
    '''Takes a string, replaces underscores with spaces, and capitalizes the first letter of each word'''
    def capitalize(s):
        return s[0].upper() + s[1:]

    words = s.split("_")
    return " ".join([capitalize(word) for word in words])

def display_training_history(history):
    '''Plot training & validation loss and metric values'''
    prop_categories = {}
    for prop in history.history:
        prop_type, prop_category = split_term(prop)
        # value to store
        storage_value = (prop_type, prop)
        try:
            prop_categories[prop_category].append(storage_value)
        except KeyError:
            prop_categories[prop_category] = [storage_value]

    plot_dims = get_square_factors(len(prop_categories))
    ordered_categories = sorted([category for category in prop_categories])

    for category_index, category in enumerate(ordered_categories):
        plt.subplot(plot_dims[0], plot_dims[1], category_index + 1)
        legend_props = []
        nice_category = title_term(category)
        plt.title(f"{nice_category} Metrics")
        plt.ylabel(nice_category)
        plt.xlabel("Epoch")
        category_values = prop_categories[category]
        for prop_values in category_values:
            prop_type, prop = prop_values
            plt.plot(history.history[prop])
            legend_props.append(prop_type)
        plt.legend(legend_props, loc=('upper right' if nice_category == title_term("loss") else 'upper left'))

    plt_show()
