from copy import deepcopy

def format(element):
    return [element, [deepcopy(element)]]

def sum_int(a: int, b:int):
    return a + b

def sub_int(a: int, b:int):
    return a - b

def sum_interval(interval1: int, interval2: int):
    return max(interval1, interval2)

def sub_interval(interval: int, interval_list: int):
    for i in interval_list:
        if i != 1:
            return interval
    return 1

def sum_sums(list1: list, list2: list):
    return [w1 + w2 for w1, w2 in zip(list1, list2)]

def sub_sums(list1: list, list2: list):
    return [w1 - w2 for w1, w2 in zip(list1, list2)]

def sum_bars(list1: list, list2: list):
    merged_list = []
    i = j = 0

    while i < len(list1) and j < len(list2):
        if list1[i][0] < list2[j][0]:
            merged_list.append(list1[i])
            i += 1
        elif list1[i][0] == list2[j][0]:
            merged_list.append([list1[i][0], list1[i][1] + list2[j][1]])
            i += 1
            j += 1
        else:
            merged_list.append(list2[j])
            j += 1

    while i < len(list1):
        merged_list.append(list1[i])
        i += 1

    while j < len(list2):
        merged_list.append(list2[j])
        j += 1

    return merged_list

def sub_bars(list1, list2):
    result = []
    i = j = 0

    while i < len(list1) and j < len(list2):
        if list1[i][0] < list2[j][0]:
            result.append(list1[i])
            i += 1
        elif list1[i][0] == list2[j][0]:
            quantity = list1[i][1]-list2[j][1]
            if quantity:
                result.append([list1[i][0], quantity])
            i += 1
            j += 1
        else:
            j += 1

    while i < len(list1):
        result.append(list1[i])
        i += 1
    return result

def sort_and_group(list1: list, name):
    if not len(list1):
        return []
    list1.sort()
    anterior = list1.pop(0)
    if name != "AREAS":
        anterior = int(anterior)
    list2 = [[anterior, 1]]
    for value in list1:
        if name != "AREAS":
            value = int(value)
        if value != anterior:
            list2.append([value, 1])
            anterior = value
        else:
            list2[-1][1] += 1
    return list2

def group_in_intervals(bars, interval):
    new_list = []
    for messure, q in bars:
        new_messure = round(messure/interval)*interval
        if not len(new_list) or new_messure > new_list[-1][0]:
            new_list.append([new_messure, q])
        else:
            new_list[-1][1] += q
    return new_list