import csv
from csv import DictWriter
import phonword_python3 as phonword
from wuggy import WuggyGenerator
from g2p_en import G2p 
from scipy import spatial
#phon =

#print(phonword.dl("cat", "television"))
#print(phonword.dlNormalized("cat", "television"))
#make 1 table with real word,
#to add lev dists to table, just add lev and norm lev to tuples
'''
g = WuggyGenerator()
g.load("orthographic_english")
for match in g.generate_classic(["rough"], ncandidates_per_sequence=100):
    print(match["pseudoword"])


print(phonword.getPhonemes("furl"))
print(len([['chair']]))
myWordGenPCS_c_o = phonword.PhonWordRepStress(d = 1024, ngramType = "o", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "c")
valueVec1 = myWordGenPCS_c_o.make_repPhonStress(['CH', 'EH1', 'R'])
print(valueVec1)
valueVec2 = myWordGenPCS_c_o.make_repPhonStress('chair')
print(valueVec2)
print(1 - spatial.distance.cosine(valueVec1, valueVec2))
'''
#not found in lexicon orthographic_english: mold, burl
#not in CMU pronouncing dictionary: furl, leer

def maxCos(lst):
    if len(lst) == 0:
        return ""
    max = lst[0]
    for i in range(len(lst)):
        if lst[i][1] > max[1]:
            max = lst[i]
    return max[0]   

def leastDL(lst):
    if len(lst) == 0:
        return ""
    least = lst[0]
    for i in range(len(lst)):
        if lst[i][2] < least[2]:
            least = lst[i]
    return least[0]   

def leastDLNorm(lst):
    if len(lst) == 0:
        return ""
    least = lst[0]
    for i in range(len(lst)):
        if lst[i][3] < least[3]:
            least = lst[i]
    return least[0]  

def leastDLPhon(lst):
    if len(lst) == 0:
        return ""
    least = lst[0]
    print(lst)
    for i in range(len(lst)):
        if lst[i][4] < least[4]:
            least = lst[i]
    return least[0]   

def top10DLPhon(lst):
    if len(lst) < 10:
        return []
    lst = sorted(lst, key=lambda x: x[1])
    return get_first_elements(lst[0:(len(lst))])

def leastDLNormPhon(lst):
    if len(lst) == 0:
        return ""
    least = lst[0]
    for i in range(len(lst)):
        if lst[i][5] < least[5]:
            least = lst[i]
    return least[0]  

def get_first_elements(tuple_list):
    return [x[0] for x in tuple_list]

def capital(lst):
    for i in range(len(lst)):
        print(lst[i])
        lst[i] = lst[i].upper()
    return lst



with open(r'C:\\Users\\Stephen\\Downloads\\DRM_Words_Pronunciations - PhonologicalDRM.csv', 'r') as realWords:
    with open(r'pseudowordOutputNew.csv', 'w', newline='') as output:
        myWordGenPCS_c_o = phonword.PhonWordRepStress(d = 1024, ngramType = "o", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "c")
        phonDict = {}
        g2p = G2p()
        w = csv.writer(output)
        field = ["real word", "pesudoword1", "pesudoword2", "pesudoword3", "pesudoword4", "pesudoword5", "pesudoword6",	"pesudoword7", "pesudoword8", "pesudoword9", "pesudoword10"]
        w.writerow(field)
        g = WuggyGenerator()
        g.load("orthographic_english")
        targets = ["chair", "bread", "cold", "flag", "girl", "river", "rough", "sleep", "slow", "smell", "soft", "sweet", "thief", "trash", "beer", "black", "dog", "rain", "right", "sick", "hand", "law"]
        targets = ["cold", "rough", "slow", "smell", "trash", "black", "dog", "right"]
        for target in targets:
            pseudos = []
            for match in g.generate_classic([target], ncandidates_per_sequence = 1000):
                matches = match["pseudoword"]
                if g2p(target[-1]) == g2p(matches[-1]) and len(g2p(matches)) > 1:
                    pseudos = pseudos + [match["pseudoword"]]
            if (len(pseudos) < 10):
                print(target, pseudos)
                
            #w.writerow([target] + pseudos)
            pseudoPhons = pseudos
            for i in range(len(pseudoPhons)):
                pseudoPhons[i] = (pseudoPhons[i], g2p(pseudoPhons[i]))
            phonDict[target] = pseudoPhons
        for key in phonDict:
            pseudoCossims = []
            for value in phonDict[key]:
                print(key, value)
                pseudoCossims = pseudoCossims + [(value[0], (abs(phonword.dlNormalized(phonword.getPhonemesStress(key), value[1]))))]
            #print(pseudoCossims)
            keyList = [key]
            top10 = top10DLPhon(pseudoCossims)
            keyList.extend(top10)
            w.writerow(keyList)

         




