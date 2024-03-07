import re

def compare(a, b):
    '''Fuzzy matching of strings to compare headers/footers in neighboring pages'''

    count = 0
    a = re.sub('\d', '@', a)
    b = re.sub('\d', '@', b)
    for x, y in zip(a, b):
        if x == y:
            count += 1
    return count / max(len(a), len(b))