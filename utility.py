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
    result = []
    for i in xrange(len(vec1)):
        result.append(vec1[i] - vec2[i])
    return result

def vector_add(vec1, vec2):
    result = []
    for i in xrange(len(vec1)):
        result.append(vec1[i] + vec2[i])
    return result