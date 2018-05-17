class Example:
    # input should be a integer array of length 28*28
    # label should be index of english letter (e.g. a=0, b=1, ...)
    def __init__(self, input, label, label_vector_len = 10):
        self.input = input
        self.label = label

        self.label_vector = [0] * label_vector_len
        self.label_vector[label] = 1
