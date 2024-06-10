def sum_int(a: int, b:int):
    return a + b

def sub_int(a: int, b:int):
    return a - b

def sum_sums(list1: list, list2: list):
    return [w1 + w2 for w1, w2 in zip(list1, list2)]

def sub_sums(list1: list, list2: list):
    return [w1 - w2 for w1, w2 in zip(list1, list2)]

def choose_interval(list1, second_interval, i):
    batch_size = 60
    maximum_batches = 5

    chosen_option = "self"

    if not i%batch_size:
        list1[1].append([0])

    if i == batch_size*maximum_batches:
        list1[1].pop(0)
        max_interval = max(list1[1])
        list1[0] = max_interval

    if second_interval > list1[0]:
        list1[0] = second_interval
        chosen_option = "other"
    
    list1[1][-1] = max(list1[1][-1], second_interval)
    
    return list1, chosen_option

def sum_bars(list1: list, list2: list):
    merged_list = []
    i = j = 0

    while i < len(list1) and j < len(list2):
        if list1[i][0] < list2[j][0]:
            merged_list.append(list1[i])
            i += 1
        elif list1[i][0] == list2[j][0]:
            merged_list.append([list1[i][0], list1[i][1] + list2[i][1]])
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
            result.append([list1[i][0], list1[i][1]-list1[j][1]])
            i += 1
            j += 1
        else:
            j += 1

    while i < len(list1):
        result.append(list1[i])
        i += 1

    return result

def sum_area_bars(list1: list, list2:list, interval):
    merged_list = []
    i = j = 0

    interval *= 0.5
    while i < len(list1) and j < len(list2):
        if list1[i][0] + interval < list2[j][0]:
            merged_list.append(list1[i])
            i += 1
        elif list1[i][0] - interval <= list2[j][0] < list1[i][0] + interval:
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

def sub_area_bars(list1, list2, interval):
    result = []
    i = j = 0

    interval *= 0.5
    while i < len(list1) and j < len(list2):
        if list1[i][0] + interval < list2[j][0]:
            result.append(list1[i])
            i += 1
        elif list1[i][0] - interval <= list2[j][0] < list1[i][0] + interval:
            result.append([list1[i][0], list1[i][1]-list2[j][1]])
            j += 1
            j += 1
        else:
            j += 1

    while i < len(list1):
        result.append(list1[i])
        i += 1

    return result

def sort_and_group(list1: list):
    # sorts and groups
    if not len(list1):
        return []
    list1.sort()
    list2 = [[list1.pop(0), 1]]
    anterior = list1[0]
    for value in list1[1:]:
        if value != anterior:
            list2.append([value, 1])
            anterior = value
        else:
            list2[-1][1] += 1
    return list2