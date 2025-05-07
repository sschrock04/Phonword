import phonword_python3 as phonword
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

numSims = 1

#import phonological similarity rating raw data
data = pd.read_csv("C:\\Users\\Stephen\\Downloads\\SiewCastro2023.csv")

#import word pairs rated
pairs = pd.read_csv("C:\\Users\\Stephen\\Downloads\\word-pairs.csv")

#get cross-participant average ratings for the 200 word pair
rating = data.groupby(['word1']).mean(numeric_only=True)

#combine word pairs and ratings
df = pd.merge(pairs, rating, how='left',left_on= ['word1'], right_on= ['word1'])


#Original orthographic rep
Holoword = {'simRating':list(), 'phonDistance':list(), 'acousticSim':list()}
#CMU transcript no stress
Phonword = {'simRating':list(), 'phonDistance':list(), 'acousticSim':list()}
#IPA transcript
IPAword = {'simRating':list(), 'phonDistance':list(), 'acousticSim':list()}
#CMU transcript with stress
PhonwordStress = {'simRating':list(), 'phonDistance':list(), 'acousticSim':list()}
#Combined Holoword, Phonword sum vector
HPSum = {'simRating':list(), 'phonDistance':list(), 'acousticSim':list()}
#Combined Holoword, IPAword sum vector
HISum = {'simRating':list(), 'phonDistance':list(), 'acousticSim':list()}
#Combined Holoword, Phonword concatenated vector
HPCon = {'simRating':list(), 'phonDistance':list(), 'acousticSim':list()}
#Combined Holoword, IPAword concatenated vector
HICon = {'simRating':list(), 'phonDistance':list(), 'acousticSim':list()}

Phonworda = {'simRating':list(), 'phonDistance':list(), 'acousticSim':list()}
Phonwordb = {'simRating':list(), 'phonDistance':list(), 'acousticSim':list()}
PhonwordIa = {'simRating':list(), 'phonDistance':list(), 'acousticSim':list()}
PhonwordIb = {'simRating':list(), 'phonDistance':list(), 'acousticSim':list()}

for i in range(numSims):
    if (i % 20 == 0):
        print(i)
    #create a "word-form vector generator"
    myWordGen = phonword.PhonWordRep(d = 1024, ngramType = "o", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "convolution")
    
    myWordGenI = phonword.IPAWordRep(d = 1024, ngramType = "o", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "convolution")
    myWordGenH = phonword.HoloWordRep(d = 1024, ngramType = "o", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "convolution")
    myWordGenStress = phonword.PhonWordRepStress(d = 1024, ngramType = "tr", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "convolution")

    myWordGena = phonword.PhonWordRep(d = 1024, ngramType = "tr", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "a")
    myWordGenb = phonword.PhonWordRep(d = 1024, ngramType = "tr", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "b")
    myWordGenIa = phonword.PhonWordRep(d = 1024, ngramType = "tr", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "a")
    myWordGenIb = phonword.PhonWordRep(d = 1024, ngramType = "tr", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "b")

    
    cossim_list = list()
    cossim_listI = list()
    cossim_listH = list()
    cossim_listStress = list()
    cossim_listHPSum = list()
    cossim_listHISum = list()
    cossim_listHPCon = list()
    cossim_listHICon = list()

    cossim_lista = list()
    cossim_listb = list()
    cossim_listIa = list()
    cossim_listIb = list()
    for i in range(0, len(df)):
        #calculate cosine similarity between representation vector of each word pair, then add cosine similarity to list
        w1vec = myWordGen.make_repPhon(df['word1'][i])
        w2vec = myWordGen.make_repPhon(df['word2'][i])
        phonword.euclidean_distance(w1vec, w2vec)
        cossim = 1 - spatial.distance.cosine(w1vec, w2vec)
        cossim_list.append(abs(cossim))
    

        w1vecI = myWordGenI.make_repPhon(df['word1'][i])
        w2vecI = myWordGenI.make_repPhon(df['word2'][i])
        phonword.euclidean_distance(w1vec, w2vec)
        cossimI = 1 - spatial.distance.cosine(w1vecI, w2vecI)
        cossim_listI.append(abs(cossimI))

        w1vecH = myWordGenH.make_rep(df['word1'][i])
        w2vecH = myWordGenH.make_rep(df['word2'][i])
        phonword.euclidean_distance(w1vec, w2vec)
        cossimH = 1 - spatial.distance.cosine(w1vecH, w2vecH)
        cossim_listH.append(abs(cossimH))

        w1vecStress = myWordGenStress.make_repPhonStress(df['word1'][i])
        w2vecStress = myWordGenStress.make_repPhonStress(df['word2'][i])
        phonword.euclidean_distance(w1vec, w2vec)
        cossimStress = 1 - spatial.distance.cosine(w1vecStress, w2vecStress)
        cossim_listStress.append(abs(cossimStress))

        w1vecHPSum = w1vec + w1vecH
        w2vecHPSum = w2vec + w2vecH
        phonword.euclidean_distance(w1vec, w2vec)
        cossimHPSum = 1 - spatial.distance.cosine(w1vecHPSum, w2vecHPSum)
        cossim_listHPSum.append(abs(cossimHPSum))
        
        w1vecHISum = w1vecI + w1vecH
        w2vecHISum = w2vecI + w2vecH
        phonword.euclidean_distance(w1vec, w2vec)
        cossimHISum = 1 - spatial.distance.cosine(w1vecHISum, w2vecHISum)
        cossim_listHISum.append(abs(cossimHISum))

        w1vecHPCon = np.concatenate((w1vec, w1vecH))
        w2vecHPCon = np.concatenate((w2vec, w2vecH))
        phonword.euclidean_distance(w1vec, w2vec)
        cossimHPCon = 1 - spatial.distance.cosine(w1vecHPCon, w2vecHPCon)
        cossim_listHPCon.append(abs(cossimHPCon))
        
        w1vecHICon = np.concatenate((w1vecI, w1vecH))
        w2vecHICon = np.concatenate((w2vecI, w2vecH))
        phonword.euclidean_distance(w1vec, w2vec)
        cossimHICon = 1 - spatial.distance.cosine(w1vecHICon, w2vecHICon)
        cossim_listHICon.append(abs(cossimHICon))

        w1veca = myWordGena.make_repPhon(df['word1'][i])
        w2veca = myWordGena.make_repPhon(df['word2'][i])
        phonword.euclidean_distance(w1vec, w2vec)
        cossima = 1 - spatial.distance.cosine(w1veca, w2veca)
        cossim_lista.append(abs(cossima))

        w1vecb = myWordGenb.make_repPhon(df['word1'][i])
        w2vecb = myWordGenb.make_repPhon(df['word2'][i])
        phonword.euclidean_distance(w1vec, w2vec)
        cossimb = 1 - spatial.distance.cosine(w1vecb, w2vecb)
        cossim_listb.append(abs(cossimb))

        w1vecIa = myWordGenIa.make_repPhon(df['word1'][i])
        w2vecIa = myWordGenIa.make_repPhon(df['word2'][i])
        phonword.euclidean_distance(w1vec, w2vec)
        cossimIa = 1 - spatial.distance.cosine(w1vecIa, w2vecIa)
        cossim_listIa.append(abs(cossimIa))

        w1vecIb = myWordGenIb.make_repPhon(df['word1'][i])
        w2vecIb = myWordGenIb.make_repPhon(df['word2'][i])
        phonword.euclidean_distance(w1vec, w2vec)
        cossimIb = 1 - spatial.distance.cosine(w1vecIb, w2vecIb)
        cossim_listIb.append(abs(cossimIb))
        
    #add lists of cosine similarities to df
    df['cossim'] = cossim_list
    df['cossimIPA'] = cossim_listI
    df['cossimHolo'] = cossim_listH
    df['cossimStress'] = cossim_listStress
    df['cossimHPSum'] = cossim_listHPSum
    df['cossimHISum'] = cossim_listHISum
    df['cossimHPCon'] = cossim_listHPCon
    df['cossimHICon'] = cossim_listHICon

    df['cossima'] = cossim_lista
    df['cossimb'] = cossim_listb
    df['cossimIa'] = cossim_listIa
    df['cossimIb'] = cossim_listIb

    #add correlation between each cosine similarity list and testing parameter to each dictionary
    Holoword['simRating'].append(df["Response"].corr(df["cossimHolo"]))
    Holoword['phonDistance'].append(df["distance_x"].corr(df["cossimHolo"]))
    Holoword['acousticSim'].append(df["acoustic_sim"].corr(df["cossimHolo"]))

    Phonword['simRating'].append(df["Response"].corr(df["cossim"]))
    Phonword['phonDistance'].append(df["distance_x"].corr(df["cossim"]))
    Phonword['acousticSim'].append(df["acoustic_sim"].corr(df["cossim"]))

    IPAword['simRating'].append(df["Response"].corr(df["cossimIPA"]))
    IPAword['phonDistance'].append(df["distance_x"].corr(df["cossimIPA"]))
    IPAword['acousticSim'].append(df["acoustic_sim"].corr(df["cossimIPA"]))

    PhonwordStress['simRating'].append(df["Response"].corr(df["cossimStress"]))
    PhonwordStress['phonDistance'].append(df["distance_x"].corr(df["cossimStress"]))
    PhonwordStress['acousticSim'].append(df["acoustic_sim"].corr(df["cossimStress"]))

    HPSum['simRating'].append(df["Response"].corr(df["cossimHPSum"]))
    HPSum['phonDistance'].append(df["distance_x"].corr(df["cossimHPSum"]))
    HPSum['acousticSim'].append(df["acoustic_sim"].corr(df["cossimHPSum"]))

    HISum['simRating'].append(df["Response"].corr(df["cossimHISum"]))
    HISum['phonDistance'].append(df["distance_x"].corr(df["cossimHISum"]))
    HISum['acousticSim'].append(df["acoustic_sim"].corr(df["cossimHISum"]))

    HPCon['simRating'].append(df["Response"].corr(df["cossimHPCon"]))
    HPCon['phonDistance'].append(df["distance_x"].corr(df["cossimHPCon"]))
    HPCon['acousticSim'].append(df["acoustic_sim"].corr(df["cossimHPCon"]))

    HICon['simRating'].append(df["Response"].corr(df["cossimHICon"]))
    HICon['phonDistance'].append(df["distance_x"].corr(df["cossimHICon"]))
    HICon['acousticSim'].append(df["acoustic_sim"].corr(df["cossimHICon"]))

    Phonworda['simRating'].append(df["Response"].corr(df["cossima"]))
    Phonworda['phonDistance'].append(df["distance_x"].corr(df["cossima"]))
    Phonworda['acousticSim'].append(df["acoustic_sim"].corr(df["cossima"]))

    Phonwordb['simRating'].append(df["Response"].corr(df["cossimb"]))
    Phonwordb['phonDistance'].append(df["distance_x"].corr(df["cossimb"]))
    Phonwordb['acousticSim'].append(df["acoustic_sim"].corr(df["cossimb"]))

    PhonwordIa['simRating'].append(df["Response"].corr(df["cossimIa"]))
    PhonwordIa['phonDistance'].append(df["distance_x"].corr(df["cossimIa"]))
    PhonwordIa['acousticSim'].append(df["acoustic_sim"].corr(df["cossimIa"]))

    PhonwordIb['simRating'].append(df["Response"].corr(df["cossimIb"]))
    PhonwordIb['phonDistance'].append(df["distance_x"].corr(df["cossimIb"]))
    PhonwordIb['acousticSim'].append(df["acoustic_sim"].corr(df["cossimIb"]))


#take the averages of the correlations
for key in Holoword:
    Holoword[key] = np.mean(Holoword[key])

for key in Phonword:
    Phonword[key] = np.mean(Phonword[key])

for key in IPAword:
    IPAword[key] = np.mean(IPAword[key])

for key in PhonwordStress:
    PhonwordStress[key] = np.mean(PhonwordStress[key])

for key in HPSum:
    HPSum[key] = np.mean(HPSum[key])

for key in HISum:
    HISum[key] = np.mean(HISum[key])

for key in HPCon:
    HPCon[key] = np.mean(HPCon[key])

for key in HICon:
    HICon[key] = np.mean(HICon[key])

for key in Phonworda:
    Phonworda[key] = np.mean(Phonworda[key])

for key in Phonwordb:
    Phonwordb[key] = np.mean(Phonwordb[key])

for key in PhonwordIa:
    PhonwordIa[key] = np.mean(PhonwordIa[key])

for key in PhonwordIb:
    PhonwordIb[key] = np.mean(PhonwordIb[key])




#print correlations for each parameter and measure
print('Correlation data over', numSims, 'simulations')
print('Holoword: ', Holoword)
print('Phonword: ', Phonword)
print('IPAword: ', IPAword)
print('PhonwordStress: ', PhonwordStress)
print('HPSum: ', HPSum)
print('HISum: ', HISum)
print('HPCon: ', HPCon)
print('HICon: ', HICon)

print('Phonworda: ', Phonworda)
print('Phonwordb: ', Phonwordb)
print('PhonwordIa: ', PhonwordIa)
print('PhonwordIb: ', PhonwordIb)