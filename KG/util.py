import theano.tensor as T
import theano
import numpy as np
import random

def shared32(x, name=None, borrow=False):

    return theano.shared(np.asarray(x, dtype='float32'), name=name, borrow=borrow)

def computePR(tagdic, gold, predict):# gold and predict are a list of tag sequences
    

    correct = 0.
    candidate = 0.
    extract = 0.

    dic = {}

    goldens = scanEns(tagdic, gold)
    predens = scanEns(tagdic, predict)

    for preden, golden in zip(predens, goldens):

        correcten = []
        for item in preden:
            if item in golden:
                correcten.append(item)

        correct += len(correcten)
        extract += len(preden)
        candidate += len(golden)


        for item in golden:
            dic.setdefault(item[2], [0.,0.,0.]) #gold, pre, correct
            dic[item[2]][0] += 1
        for item in preden:
            dic.setdefault(item[2], [0.,0.,0.]) #gold, pre, correct
            dic[item[2]][1] += 1
        for item in correcten:
            dic.setdefault(item[2], [0.,0.,0.]) #gold, pre, correct
            dic[item[2]][2] += 1
       

    pr_type = {} # precision, recall, f1 for each entity type
    for key in dic:
        stat = dic[key]
        precision, recall = stat[2]/(stat[1] + 1e-8), stat[2]/(stat[0]+ 1e-8)
        f1 = 2./(1./(precision+1e-7)+1./(recall+1e-7))
        pr_type[key] = [precision, recall, f1]

    precision, recall = correct/(extract + 1e-8), correct/(candidate+ 1e-8)
    f1 = 2./(1./(precision+1e-7)+1./(recall+1e-7))
    overall = [precision, recall, f1]

    return overall, pr_type


#return a list of entities from a sequence of tag, each is a tuple (start, end, type)
#suited for B-I  tag encoding strategy
 
def scanEns(tagdic, taglists):

    tool = {}
    enname = set()
    for key in tagdic:
        if key != "O":
            entity = key[key.index("-")+1:]
            enname.add(entity)

    for key in enname:
        tool[tagdic["B-"+key]] = [tagdic["B-"+key], tagdic["I-"+key],key]
    
    result = []
    for tags in taglists:
        entities = []
        i = 0
        while i < len(tags):
            if tags[i] in tool:
                en = tool[tags[i]]
                k = i+1
                while k < len(tags) and tags[k] == en[1] : k = k+1
                entities.append((i, k, en[2]))
                i = k - 1

            i += 1
        result.append(entities)

    return result


def randomdata(data):
    for i in xrange(len(data) - 1):
        randindex = random.randint(i+1, len(data) - 1)
        data[i], data[randindex] = data[randindex], data[i]



def get_bestepoch(scoref):
    f = open(scoref)
    bestf = 0.
    for line in f:
        r = re.match("\D+(\d)\D+(\d\.\d+)\D+(\d\.\d+)\D+(\d\.\d+)", line)
        epoch = int(r.group(1))
        f = float(r.group(4))
        if f > bestf:
            beste = epoch
            bestf = f

    return beste





