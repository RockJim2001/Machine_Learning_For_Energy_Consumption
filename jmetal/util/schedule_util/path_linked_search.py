import random
import copy

def pathrelinked(nondominated, representive):
    (os1,ms1,ws1) = nondominated
    (os2,ms2,ws2) = representive
    length = len(os1)
    idxset = [i for i in range(length)]
    random.shuffle(idxset)
    idx1 = idxset[:length//3]
    idx2 = idxset[length//3:2*(length//3)]

    off1 = copy.copy(representive)
    off2 = copy.copy(representive)

    # OS片段
    for i in range(length//3):
        if os1[i] != os2[i]:
            loc = os2.index(os1[i])
            os2[i],os2[loc] = os2[loc],os2[i]
    off1[0] = os2

    for i in range(length//3, 2*(length//3)):
        if os1[i] != os2[i]:
            loc = os2.index(os1[i])
            os2[i],os2[loc] = os2[loc],os2[i]
    off2[0] = os2

    # MS,WS片段
    for idx in idx1:
        ws2[idx] = ws1[idx]
        ms2[idx] = ms1[idx]
    off1[1] = ms2
    off1[2] = ws2

    for idx in idx2:
        ws2[idx] = ws1[idx]
        ms2[idx] = ms1[idx]
    off2[1] = ms2
    off2[2] = ws2

    return off1, off2