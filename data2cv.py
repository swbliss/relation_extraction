import cPickle
import math
import time

from dataset import *


# Get new sentence with zero padding
def get_idx(sentence, filter_h=5, max_l=100):
    pad = int(math.ceil(filter_h / 2.0))
    x = [0] * pad
    if len(sentence) < max_l:
        for ind in sentence:
            x.append(ind)
    else:
        for i in xrange(max_l):
            x.append(sentence[i])
    # padding the end of sentence
    while len(x) < max_l + 2 * pad:
        x.append(0)
    return x


# TODO(swjung): check what is the purpose of this function
def get_pf(sentLen, allIndice, filter_h=5, max_l=100):
    if sentLen < max_l:
        index = np.arange(sentLen)
    else:
        index = np.arange(max_l)

    sent_pf1 = index - allIndice[0] + (max_l - 1)
    sent_pf2 = index - allIndice[1] + (max_l - 1)

    # padding the begining of sentence
    pad = int(math.ceil(filter_h / 2.0))
    pf1 = [-1] * pad
    pf2 = [-1] * pad

    pf1.extend(sent_pf1)
    pf2.extend(sent_pf2)

    # padding the end of sentence
    while len(pf1) < max_l + 2 * pad:
        pf1.append(-1)
        pf2.append(-1)
    return [pf1, pf2]


# ouput: list of InstnaceBag
# InstanceBag: bag of instances in the form (entities, rels, num,...) related with one entity pair
def make_idx_data_cv(data, filter_h, max_l):
    newData = []
    for ins in data:
        entities = ins.entities
        rel = ins.rel
        num = ins.num
        sentences = ins.sentences
        positions = ins.positions
        entitiesPos = ins.entitiesPos
        newSent = []
        newPos = []
        newEPos = []

        # Handling some weird entity positions
        remove_idx = []
        for i, pos in enumerate(entitiesPos):
            if pos[0] > max_l - 1 or pos[1] > max_l - 1:
                remove_idx.append(i)
                num -= 1

        for i, sentence in enumerate(sentences):
            if i in remove_idx:
                continue
            idx = get_idx(sentence, filter_h, max_l)
            newSent.append(idx)
            pf = get_pf(len(sentence), positions[i], filter_h, max_l)
            newPos.append(pf)
            newEPos.append(entitiesPos[i])
        if len(newSent) == 0:
            continue
        newIns = InstanceBag(entities, rel, num, newSent, newPos, newEPos)
        newData += [newIns]

    return newData


if __name__ == "__main__":
    print "load test and train raw data..."
    testData = cPickle.load(open('test_len_60_gap_40.p'))
    trainData = cPickle.load(open('test_len_60_gap_40.p'))

    sentence_len = 60
    max_filter_len = 3
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print 'point 0 time: ' + '\t\t' + str(now)
    test = make_idx_data_cv(testData, max_filter_len, sentence_len)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print 'point 1 time: ' + '\t\t' + str(now)
    train = make_idx_data_cv(trainData, max_filter_len, sentence_len)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print 'point 2 time: ' + '\t\t' + str(now)

    f = open('test_3_60.p', 'w')
    cPickle.dump(test, f, -1)
    f.close()

    f = open('train_3_60.p', 'w')
    cPickle.dump(train, f, -1)
    f.close()
