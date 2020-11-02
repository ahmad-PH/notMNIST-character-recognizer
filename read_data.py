import glob
from scipy import misc
import random

from utility import *
from Example import *

test_ratio = 0.2
def read_data(max = None):
    result = []
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    max_images_per_letter = float("+inf")
    if max != None:
        max_images_per_letter = max / len(letters)

    for i, letter in enumerate(letters):

        num_images_read_for_letter = 0
        for image_path in glob.glob("../data/" + letter + "/*.png"):

            img = misc.imread(name=image_path).flatten().tolist()
            vector_div_inplace(img, 128)
            result.append(Example(img, i, 10))

            num_images_read_for_letter += 1
            if num_images_read_for_letter >= max_images_per_letter:
                break

    random.shuffle(result)

    break_index = int(len(result) * test_ratio)

    test_data = result[0:break_index]
    train_data = result[break_index+1:]

    return train_data, test_data

