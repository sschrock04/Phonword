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
    for i in range(len(lst)):
        if lst[i][4] < least[4]:
            least = lst[i]
    return least[0]   

def leastDLNormPhon(lst):
    if len(lst) == 0:
        return ""
    least = lst[0]
    for i in range(len(lst)):
        if lst[i][5] < least[5]:
            least = lst[i]
    return least[0]  

def capital(lst):
    for i in range(len(lst)):
        print(lst[i])
        lst[i] = lst[i].upper()
    return lst



with open(r'C:\\Users\\Stephen\\Downloads\\DRM_Words_Pronunciations - PhonologicalDRM.csv', 'r') as realWords:
    with open(r'pseudowordOutput.csv', 'w', newline='') as output:
        myWordGenPCS_c_o = phonword.PhonWordRepStress(d = 1024, ngramType = "o", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "c")
        phonDict = {}
        g2p = G2p()
        w = csv.writer(output)
        field = ["real word", "cosine sim", "orthographic levenshtein dist", "orthographic levenshtein dist norm", "phonological levenshtein dist", "phonological levenshtein dist norm"]
        w.writerow(field)
        g = WuggyGenerator()
        g.load("orthographic_english")
        for line1 in realWords:
            pseudos = []
            line1 = line1.strip('\n')
            for match in g.generate_classic([line1], ncandidates_per_sequence = 20):
                matches = match["pseudoword"]
                if g2p(line1[-1]) == g2p(matches[-1]) and len(g2p(matches)) > 1:
                    pseudos = pseudos + [match["pseudoword"]]
            if (len(pseudos) == 1 or len(pseudos) == 0):
                print(line1, pseudos)
                
            #w.writerow([line1] + pseudos)
            pseudoPhons = pseudos
            for i in range(len(pseudoPhons)):
                pseudoPhons[i] = (pseudoPhons[i], g2p(pseudoPhons[i]))
            phonDict[line1] = pseudoPhons
        for key in phonDict:
            keyVec = myWordGenPCS_c_o.make_repPhonStress(key)
            pseudoCossims = []
            for value in phonDict[key]:
                print(key, value)
                valueVec = myWordGenPCS_c_o.make_repPhonStress(value[1])
                cossim = 1 - spatial.distance.cosine(keyVec, valueVec)
                pseudoCossims = pseudoCossims + [(value[0], cossim, phonword.dl(key, value[0]), phonword.dlNormalized(key, value[0]), phonword.dl(phonword.getPhonemesStress(key), value[1]), phonword.dlNormalized(phonword.getPhonemesStress(key), value[1]))]
            #print(pseudoCossims)
            w.writerow([key] + [maxCos(pseudoCossims)] + [leastDL(pseudoCossims)] + [leastDLNorm(pseudoCossims)] + [leastDLPhon(pseudoCossims)] + [leastDLNormPhon(pseudoCossims)])

         





