def sum_dictionarys(dict1: dict, dict2: dict) -> dict:
    for key in dict1:
        if dict2.get(key, -1) > 0:
            dict1[key] += dict2[key]
            dict2.pop(key)
    for key in dict2:
        if dict1.get(key, -1) == -1:
            dict1[key] = dict2[key]
    return dict1

