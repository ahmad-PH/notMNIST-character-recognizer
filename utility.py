import datetime

def max(list):
    max_index = -1;
    max_value = float("-inf")
    for i, item in enumerate(list):
        if (item > max_value):
            max_value = item
            max_index = i
    return (max_index, max_value)


class Timer:
    def __int__(self):
        self.time = None

    def record(self):
        self.time = datetime.datetime.now()

    def print_elapsed(self, message = ""):
        print message, (datetime.datetime.now() - self.time)

def vector_subtract(vec1, vec2):
    if len(vec1) != len(vec2):
        raise RuntimeError("unequal vector length at vector_subtract: " + str(len(vec1)) + ", " + str(len(vec2)))

    result = []
    for i in xrange(len(vec1)):
        result.append(vec1[i] - vec2[i])
    return result

def vector_add(vec1, vec2):
    result = []
    for i in xrange(len(vec1)):
        result.append(vec1[i] + vec2[i])
    return result

def vector_div_inplace(vec, divisor):
    for i, item in enumerate(vec):
        vec[i] = vec[i] / divisor


def vector_power_two(vec):
    result = 0
    for item in vec:
        result += item ** 2
    return result

def display_image(img):
    pixel_index = 0
    for i in xrange(28):
        for j in xrange(28):
            print img[pixel_index],
            pixel_index += 1
        print ""

def ask_for_user_input(message, valid_options = None):
    while True:
        user_input = raw_input(message + " ")
        if valid_options is not None and user_input not in valid_options:
            print "please type ", "/".join(valid_options)
        else:
            return user_input
