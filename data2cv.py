from dataset import *
import time
import cPickle
import math

# Get new sentence with zero padding
# TODO(swjung): This zero padding method is different from paper's implementation a bit. Follow the paper's method.
def get_idx(sentence, filter_h=5, max_l=100):

    pad = int(math.ceil(filter_h/2.0))
    x = [0]*pad
    if len(sentence) < max_l:
        for ind in sentence:
            x.append(ind)
    else:
        for i in xrange(max_l):
            x.append(sentence[i])
    #padding the end of sentence
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

# TODO(swjung): check what is the purpose of this function
def get_pf(sentLen, allIndice, filter_h=5, max_l=100):

    if sentLen < max_l:
        #index = range(0, sentLen)
        index = np.arange(sentLen)
    else:
        index = np.arange(max_l)
        #index = range(0, max_l)

    pf1 = index - allIndice[0] + 1 + 51
    pf2 = index - allIndice[1] + 1 + 51

    for i, pf in enumerate(pf1):
        if pf1[i] < 1:
            pf1[i] = 1
        # TODO(swjung): I think 101 should be replaced my (max_I + 1)
        elif pf1[i] > 101:
            pf1[i] = 101

        if pf2[i] < 1:
            pf2[i] = 1
        elif pf2[i] > 101:
            pf2[i] = 101

    pad = int(math.ceil(filter_h/2.0))
    x1 = [0]*pad
    x2 = [0]*pad

    x1.extend(pf1)
    x2.extend(pf2)
    #for i, pf in enumerate(pf1):
    #    x1.append(pf1[i])
    #    x2.append(pf2[i])

    #padding the end of sentence
    while len(x1) < max_l+2*pad:
        x1.append(0)
        x2.append(0)
    return [x1, x2]

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
        newSent = []
        newPos = []
        entitiesPos = ins.entitiesPos
        newEPos = []

        # Handling some weird entity positions
        remove_eps = []
        for i, pos in enumerate(entitiesPos):
            if pos[0] > max_l - 1 or pos[1] > max_l - 1:
                remove_eps.append(i)
                num -= 1

        for i, sentence in enumerate(sentences):
            if i in remove_eps:
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