import csv
import phonword_python3 as phonword
from g2p_en import G2p 
from wuggy import WuggyGenerator
with open(r'C:\\Users\\Stephen\\Downloads\\CastroVitevitch2022_Phonological_Association1.xlsx - Sheet 1.csv', 'r') as sheet1:
    with open(r'pseudowordOutputNew1.csv', 'w', newline='') as output:
        field = ["Source", "Target", "FAS", "No.response", "PhDiffLev", "Phonological LD", "Orthographic LD"]
        print(phonword.dl("eye", "lie"))
        w = csv.writer(output)
        w.writerow(field)
        g2p = G2p()
        g = WuggyGenerator()
        g.load("orthographic_english")
        for line in sheet1:
            line = line.split(',')
            line = line[0:5]
            if (line[0] != "Source"):
                source = phonword.getPhonemes(line[0])
                target = phonword.getPhonemes(line[1])
                if (source != "NO PHONEMES FOUND" and target != "NO PHONEMES FOUND"): 
                    w.writerow(line + [phonword.dlNormalized(source, target)] + [phonword.dlNormalized(line[0], line[1])])
                    #print("phonword", line, phonword.dlNormalized(source, target),phonword.dlNormalized(line[0], line[1]))
                else:
                    source = g2p(source)
                    target = g2p(source)
                    #print(line)
                    w.writerow(line + [phonword.dlNormalized(source, target)] + [phonword.dlNormalized(line[0], line[1])])
            


with open(r'C:\\Users\\Stephen\\Downloads\\CastroVitevitch2022_Phonological_Association3.xlsx - Sheet 1.csv', 'r') as sheet2:
    with open(r'pseudowordOutputNew2.csv', 'w', newline='') as output:
        field = ["Source", "Target", "FAS", "No.response", "PhDiffLev", "Phonological LD", "Orthographic LD"]
        w = csv.writer(output)
        w.writerow(field)
        g2p = G2p()
        g = WuggyGenerator()
        g.load("orthographic_english")
        for line in sheet2:
            line = line.split(',')
            line = line[0:5]
            if (line[0] != "Source"):
                source = phonword.getPhonemes(line[0])
                target = phonword.getPhonemes(line[1])
                if (source != "NO PHONEMES FOUND" and target != "NO PHONEMES FOUND"): 
                    w.writerow(line + [phonword.dlNormalized(source, target)] + [phonword.dlNormalized(line[0], line[1])])
                else:
                    source = g2p(source)
                    target = g2p(source)
                    #print(line)
                    w.writerow(line + [phonword.dlNormalized(source, target)] + [phonword.dlNormalized(line[0], line[1])])

