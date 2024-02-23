import ctypes
import uuid
from hashlib import sha256


def longest_string_length(lst):
    return max(len(s) for s in lst)


# return 32bit hash list
def hash_embedding(matrix, use_str=True):
    res_list = []
    for list in matrix:
        hres = 0
        for num in list:
            hres = hash((hres, num))
            hres = ctypes.c_uint32(hres).value
            if use_str:
                hres = str(hres)
        res_list.append(hres)
    return res_list


# return 256bit hash list
def hash_sentences(sentences, use_str=True):
    res_list = []
    for s in sentences:
        hres = sha256(s.encode("utf-8")).hexdigest()
        if use_str:
            hres = str(hres)
        res_list.append(hres)
    return res_list


def uuid_ids(num=1):
    return [str(uuid.uuid4()) for _ in range(num)]


def get_json_key_value(json_string="", key="description"):
    res = []
    lines = json_string.splitlines()
    # print(lines)
    for line in lines:
        if line.find(f'"{key}":') != -1:
            # print(line)
            desc = line.split(r'"')[-2]
            res.append(desc)
    return res
