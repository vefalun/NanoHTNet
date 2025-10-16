global dict_seq
dict_seq = {}

def init():
    global dict_seq
    dict_seq = {}

def setNum(key, value):
    global dict_seq
    dict_seq[key] = value

def getNum():
    global dict_seq
    return dict_seq
