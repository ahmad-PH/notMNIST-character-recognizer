def max(list):
    max_index = -1;
    max_value = float("-inf")
    for i, item in enumerate(list):
        if (item > max_value):
            max_value = item
            max_index = i
    return (max_index, max_value)