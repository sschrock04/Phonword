import csv
from csv import DictWriter
import phonword_python3 as phonword
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity



with open(r'C:\Users\Stephen\Downloads\wuggy\correlation.csv', 'w', newline='') as output:
    w = csv.writer(output)
    field = ["Model", "Human-Rated Phonological Similarity", "Phonological Distance", "Orthographic Levenshtein Distance", "Orthographic Normalized Levenshtein Distance", "Phonological Levenshtein Distance", "Phonological Normalized Levenshtein Distance"]
    w.writerow(field)
    
    numSims = 100
    #import phonological similarity rating raw data
    data = pd.read_csv("C:\\Users\\Stephen\\Downloads\\SiewCastro2023.csv")

    #import word pairs rated
    pairs = pd.read_csv("C:\\Users\\Stephen\\Downloads\\word-pairs.csv")

    #get cross-participant average ratings for the 200 word pair
    rating = data.groupby(['word1']).mean(numeric_only=True)

    #combine word pairs and ratings
    df = pd.merge(pairs, rating, how='left',left_on= ['word1'], right_on= ['word1'])

    H_c_tr = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    H_c_o = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    H_a_tr = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    H_a_o = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    H_b_tr = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    H_b_o = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    PC_c_tr = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    PC_c_o = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    PC_a_tr = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    PC_a_o = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    PC_b_tr = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    PC_b_o = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    PCS_c_tr = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    PCS_c_o = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    PCS_a_tr = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    PCS_a_o = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    PCS_b_tr = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    PCS_b_o = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    PI_c_tr = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    PI_c_o = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    PI_a_tr = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    PI_a_o = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    PI_b_tr = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    PI_b_o = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    HPC_c_tr = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    HPC_c_o = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    HPC_a_tr = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    HPC_a_o = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    HPC_b_tr = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    HPC_b_o = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    HPCS_c_tr = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    HPCS_c_o = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    HPCS_a_tr = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    HPCS_a_o = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    HPCS_b_tr = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    HPCS_b_o = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    HPI_c_tr = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    HPI_c_o = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    HPI_a_tr = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    HPI_a_o = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    HPI_b_tr = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}
    HPI_b_o = {'simRating':list(), 'phonDistance':list(), 'levDistanceOrtho':list(), 'levDistanceNormalizedOrtho':list(), 'levDistancePhon':list(), 'levDistanceNormalizedPhon':list()}

    def getCossims(list):
        ret = []
        for i in range(len(list)):
            ret.append(list[i][0])
        return ret
    
    def getlevDist(list):
        ret = []
        for i in range(len(list)):
            ret.append(list[i][1])
        return ret
    
    def getlevDistNorm(list):
        ret = []
        for i in range(len(list)):
            ret.append(list[i][2])
        return ret
    
    def getlevDistPhon(list):
        ret = []
        for i in range(len(list)):
            ret.append(list[i][3])
        return ret
    
    def getlevDistNormPhon(list):
        ret = []
        for i in range(len(list)):
            ret.append(list[i][4])
        return ret

    

    for i in range(numSims):
        if (i % 10 == 0):
            print(i)
        #create a "word-form vector generator"
        myWordGenH_c_tr = phonword.HoloWordRep(d = 1024, ngramType = "tr", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "c")
        myWordGenH_c_o = phonword.HoloWordRep(d = 1024, ngramType = "o", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "c")
        myWordGenH_a_tr = phonword.HoloWordRep(d = 1024, ngramType = "tr", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "a")
        myWordGenH_a_o = phonword.HoloWordRep(d = 1024, ngramType = "o", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "a")
        myWordGenH_b_tr = phonword.HoloWordRep(d = 1024, ngramType = "tr", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "b")
        myWordGenH_b_o = phonword.HoloWordRep(d = 1024, ngramType = "o", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "b")
        myWordGenPC_c_tr = phonword.PhonWordRep(d = 1024, ngramType = "tr", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "c")
        myWordGenPC_c_o = phonword.PhonWordRep(d = 1024, ngramType = "o", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "c")
        myWordGenPC_a_tr = phonword.PhonWordRep(d = 1024, ngramType = "tr", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "a")
        myWordGenPC_a_o = phonword.PhonWordRep(d = 1024, ngramType = "o", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "a")
        myWordGenPC_b_tr = phonword.PhonWordRep(d = 1024, ngramType = "tr", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "b")
        myWordGenPC_b_o = phonword.PhonWordRep(d = 1024, ngramType = "o", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "b")
        myWordGenPCS_c_tr = phonword.PhonWordRepStress(d = 1024, ngramType = "tr", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "c")
        myWordGenPCS_c_o = phonword.PhonWordRepStress(d = 1024, ngramType = "o", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "c")
        myWordGenPCS_a_tr = phonword.PhonWordRepStress(d = 1024, ngramType = "tr", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "a")
        myWordGenPCS_a_o = phonword.PhonWordRepStress(d = 1024, ngramType = "o", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "a")
        myWordGenPCS_b_tr = phonword.PhonWordRepStress(d = 1024, ngramType = "tr", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "b")
        myWordGenPCS_b_o = phonword.PhonWordRepStress(d = 1024, ngramType = "o", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "b")
        myWordGenPI_c_tr = phonword.IPAWordRep(d = 1024, ngramType = "tr", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "c")
        myWordGenPI_c_o = phonword.IPAWordRep(d = 1024, ngramType = "o", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "c")
        myWordGenPI_a_tr = phonword.IPAWordRep(d = 1024, ngramType = "tr", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "a")
        myWordGenPI_a_o = phonword.IPAWordRep(d = 1024, ngramType = "o", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "a")
        myWordGenPI_b_tr = phonword.IPAWordRep(d = 1024, ngramType = "tr", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "b")
        myWordGenPI_b_o = phonword.IPAWordRep(d = 1024, ngramType = "o", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "b")
        

        
        cossim_listH_c_tr = list()
        cossim_listH_c_o = list()
        cossim_listH_a_tr = list()
        cossim_listH_a_o = list()
        cossim_listH_b_tr = list()
        cossim_listH_b_o = list()
        cossim_listPC_c_tr = list()
        cossim_listPC_c_o = list()
        cossim_listPC_a_tr = list()
        cossim_listPC_a_o = list()
        cossim_listPC_b_tr = list()
        cossim_listPC_b_o = list()
        cossim_listPCS_c_tr = list()
        cossim_listPCS_c_o = list()
        cossim_listPCS_a_tr = list()
        cossim_listPCS_a_o = list()
        cossim_listPCS_b_tr = list()
        cossim_listPCS_b_o = list()
        cossim_listPI_c_tr = list()
        cossim_listPI_c_o = list()
        cossim_listPI_a_tr = list()
        cossim_listPI_a_o = list()
        cossim_listPI_b_tr = list()
        cossim_listPI_b_o = list()
        cossim_listHPC_c_tr = list()
        cossim_listHPC_c_o = list()
        cossim_listHPC_a_tr = list()
        cossim_listHPC_a_o = list()
        cossim_listHPC_b_tr = list()
        cossim_listHPC_b_o = list()
        cossim_listHPCS_c_tr = list()
        cossim_listHPCS_c_o = list()
        cossim_listHPCS_a_tr = list()
        cossim_listHPCS_a_o = list()
        cossim_listHPCS_b_tr = list()
        cossim_listHPCS_b_o = list()
        cossim_listHPI_c_tr = list()
        cossim_listHPI_c_o = list()
        cossim_listHPI_a_tr = list()
        cossim_listHPI_a_o = list()
        cossim_listHPI_b_tr = list()
        cossim_listHPI_b_o = list()
        



        for i in range(0, len(df)):
            #calculate cosine similarity between representation vector of each word pair, then add cosine similarity to list
            ld = phonword.dl(df['word1'][i], df['word2'][i])
            ldNorm = phonword.dlNormalized(df['word1'][i], df['word2'][i])

            word1PhonsPC = phonword.getPhonemes((df['word1'][i]))
            word2PhonsPC = phonword.getPhonemes((df['word2'][i]))

            word1PhonsPCS = phonword.getPhonemesStress((df['word1'][i]))
            word2PhonsPCS = phonword.getPhonemesStress((df['word2'][i]))

            word1PhonsPI = phonword.getPhonemes((df['word1'][i]))
            word2PhonsPI = phonword.getPhonemes((df['word2'][i]))
            
            ldPhon = phonword.dl(word1PhonsPC, word2PhonsPC)
            ldNormPhon = phonword.dlNormalized(word1PhonsPC, word2PhonsPC)

            w1vecH_c_tr = myWordGenH_c_tr.make_rep(df['word1'][i])
            w2vecH_c_tr = myWordGenH_c_tr.make_rep(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecH_c_tr, w2vecH_c_tr)
            cossim_listH_c_tr.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])


            w1vecH_c_o = myWordGenH_c_o.make_rep(df['word1'][i])
            w2vecH_c_o = myWordGenH_c_o.make_rep(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecH_c_o, w2vecH_c_o)
            cossim_listH_c_o.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vecH_a_tr = myWordGenH_a_tr.make_rep(df['word1'][i])
            w2vecH_a_tr = myWordGenH_a_tr.make_rep(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecH_a_tr, w2vecH_a_tr)
            cossim_listH_a_tr.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vecH_a_o = myWordGenH_a_o.make_rep(df['word1'][i])
            w2vecH_a_o = myWordGenH_a_o.make_rep(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecH_a_o, w2vecH_a_o)
            cossim_listH_a_o.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vecH_b_tr = myWordGenH_b_tr.make_rep(df['word1'][i])
            w2vecH_b_tr = myWordGenH_b_tr.make_rep(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecH_b_tr, w2vecH_b_tr)
            cossim_listH_b_tr.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vecH_b_o = myWordGenH_b_o.make_rep(df['word1'][i])
            w2vecH_b_o = myWordGenH_b_o.make_rep(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecH_b_o, w2vecH_b_o)
            cossim_listH_b_o.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vecPC_c_tr = myWordGenPC_c_tr.make_repPhon(df['word1'][i])
            w2vecPC_c_tr = myWordGenPC_c_tr.make_repPhon(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecPC_c_tr, w2vecPC_c_tr)
            cossim_listPC_c_tr.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vecPC_c_o = myWordGenPC_c_o.make_repPhon(df['word1'][i])
            w2vecPC_c_o = myWordGenPC_c_o.make_repPhon(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecPC_c_o, w2vecPC_c_o)
            cossim_listPC_c_o.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vecPC_a_tr = myWordGenPC_a_tr.make_repPhon(df['word1'][i])
            w2vecPC_a_tr = myWordGenPC_a_tr.make_repPhon(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecPC_a_tr, w2vecPC_a_tr)
            cossim_listPC_a_tr.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vecPC_a_o = myWordGenPC_a_o.make_repPhon(df['word1'][i])
            w2vecPC_a_o = myWordGenPC_a_o.make_repPhon(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecPC_a_o, w2vecPC_a_o)
            cossim_listPC_a_o.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vecPC_b_tr = myWordGenPC_b_tr.make_repPhon(df['word1'][i])
            w2vecPC_b_tr = myWordGenPC_b_tr.make_repPhon(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecPC_b_tr, w2vecPC_b_tr)
            cossim_listPC_b_tr.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vecPC_b_o = myWordGenPC_b_o.make_repPhon(df['word1'][i])
            w2vecPC_b_o = myWordGenPC_b_o.make_repPhon(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecPC_b_o, w2vecPC_b_o)
            cossim_listPC_b_o.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            ldPhon = phonword.dl(word1PhonsPCS, word2PhonsPCS)
            ldNormPhon = phonword.dlNormalized(word1PhonsPCS, word2PhonsPCS)

            w1vecPCS_c_tr = myWordGenPCS_c_tr.make_repPhonStress(df['word1'][i])
            w2vecPCS_c_tr = myWordGenPCS_c_tr.make_repPhonStress(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecPCS_c_tr, w2vecPCS_c_tr)
            cossim_listPCS_c_tr.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vecPCS_c_o = myWordGenPCS_c_o.make_repPhonStress(df['word1'][i])
            w2vecPCS_c_o = myWordGenPCS_c_o.make_repPhonStress(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecPCS_c_o, w2vecPCS_c_o)
            cossim_listPCS_c_o.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vecPCS_a_tr = myWordGenPCS_a_tr.make_repPhonStress(df['word1'][i])
            w2vecPCS_a_tr = myWordGenPCS_a_tr.make_repPhonStress(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecPCS_a_tr, w2vecPCS_a_tr)
            cossim_listPCS_a_tr.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vecPCS_a_o = myWordGenPCS_a_o.make_repPhonStress(df['word1'][i])
            w2vecPCS_a_o = myWordGenPCS_a_o.make_repPhonStress(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecPCS_a_o, w2vecPCS_a_o)
            cossim_listPCS_a_o.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vecPCS_b_tr = myWordGenPCS_b_tr.make_repPhonStress(df['word1'][i])
            w2vecPCS_b_tr = myWordGenPCS_b_tr.make_repPhonStress(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecPCS_b_tr, w2vecPCS_b_tr)
            cossim_listPCS_b_tr.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vecPCS_b_o = myWordGenPCS_b_o.make_repPhonStress(df['word1'][i])
            w2vecPCS_b_o = myWordGenPCS_b_o.make_repPhonStress(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecPCS_b_o, w2vecPCS_b_o)
            cossim_listPCS_b_o.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            ldPhon = phonword.dl(word1PhonsPI, word2PhonsPI)
            ldNormPhon = phonword.dlNormalized(word1PhonsPI, word2PhonsPI)

            w1vecPI_c_tr = myWordGenPI_c_tr.make_repPhon(df['word1'][i])
            w2vecPI_c_tr = myWordGenPI_c_tr.make_repPhon(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecPI_c_tr, w2vecPI_c_tr)
            cossim_listPI_c_tr.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vecPI_c_o = myWordGenPI_c_o.make_repPhon(df['word1'][i])
            w2vecPI_c_o = myWordGenPI_c_o.make_repPhon(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecPI_c_o, w2vecPI_c_o)
            cossim_listPI_c_o.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vecPI_a_tr = myWordGenPI_a_tr.make_repPhon(df['word1'][i])
            w2vecPI_a_tr = myWordGenPI_a_tr.make_repPhon(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecPI_a_tr, w2vecPI_a_tr)
            cossim_listPI_a_tr.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vecPI_a_o = myWordGenPI_a_o.make_repPhon(df['word1'][i])
            w2vecPI_a_o = myWordGenPI_a_o.make_repPhon(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecPI_a_o, w2vecPI_a_o)
            cossim_listPI_a_o.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vecPI_b_tr = myWordGenPI_b_tr.make_repPhon(df['word1'][i])
            w2vecPI_b_tr = myWordGenPI_b_tr.make_repPhon(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecPI_b_tr, w2vecPI_b_tr)
            cossim_listPI_b_tr.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vecPI_b_o = myWordGenPI_b_o.make_repPhon(df['word1'][i])
            w2vecPI_b_o = myWordGenPI_b_o.make_repPhon(df['word2'][i])
            cossim = 1 - spatial.distance.cosine(w1vecPI_b_o, w2vecPI_b_o)
            cossim_listPI_b_o.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            ldPhon = phonword.dl(word1PhonsPC, word2PhonsPC)
            ldNormPhon = phonword.dlNormalized(word1PhonsPC, word2PhonsPC)

            w1vec = np.concatenate((w1vecH_c_tr, w1vecPC_c_tr))
            w2vec = np.concatenate((w2vecH_c_tr, w2vecPC_c_tr))
            cossim = 1 - spatial.distance.cosine(w1vec, w2vec)
            cossim_listHPC_c_tr.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vec = np.concatenate((w1vecH_c_o, w1vecPC_c_o))
            w2vec = np.concatenate((w2vecH_c_o, w2vecPC_c_o))
            cossim = 1 - spatial.distance.cosine(w1vec, w2vec)
            cossim_listHPC_c_o.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vec = np.concatenate((w1vecH_a_tr, w1vecPC_a_tr))
            w2vec = np.concatenate((w2vecH_a_tr, w2vecPC_a_tr))
            cossim = 1 - spatial.distance.cosine(w1vec, w2vec)
            cossim_listHPC_a_tr.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vec = np.concatenate((w1vecH_a_o, w1vecPC_a_o))
            w2vec = np.concatenate((w2vecH_a_o, w2vecPC_a_o))
            cossim = 1 - spatial.distance.cosine(w1vec, w2vec)
            cossim_listHPC_a_o.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vec = np.concatenate((w1vecH_b_tr, w1vecPC_b_tr))
            w2vec = np.concatenate((w2vecH_b_tr, w2vecPC_b_tr))
            cossim = 1 - spatial.distance.cosine(w1vec, w2vec)
            cossim_listHPC_b_tr.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vec = np.concatenate((w1vecH_b_o, w1vecPC_b_o))
            w2vec = np.concatenate((w2vecH_b_o, w2vecPC_b_o))
            cossim = 1 - spatial.distance.cosine(w1vec, w2vec)
            cossim_listHPC_b_o.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            ldPhon = phonword.dl(word1PhonsPCS, word2PhonsPCS)
            ldNormPhon = phonword.dlNormalized(word1PhonsPCS, word2PhonsPCS)

            w1vec = np.concatenate((w1vecH_c_tr, w1vecPCS_c_tr))
            w2vec = np.concatenate((w2vecH_c_tr, w2vecPCS_c_tr))
            cossim = 1 - spatial.distance.cosine(w1vec, w2vec)
            cossim_listHPCS_c_tr.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vec = np.concatenate((w1vecH_c_o, w1vecPCS_c_o))
            w2vec = np.concatenate((w2vecH_c_o, w2vecPCS_c_o))
            cossim = 1 - spatial.distance.cosine(w1vec, w2vec)
            cossim_listHPCS_c_o.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vec = np.concatenate((w1vecH_a_tr, w1vecPCS_a_tr))
            w2vec = np.concatenate((w2vecH_a_tr, w2vecPCS_a_tr))
            cossim = 1 - spatial.distance.cosine(w1vec, w2vec)
            cossim_listHPCS_a_tr.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vec = np.concatenate((w1vecH_a_o, w1vecPCS_a_o))
            w2vec = np.concatenate((w2vecH_a_o, w2vecPCS_a_o))
            cossim = 1 - spatial.distance.cosine(w1vec, w2vec)
            cossim_listHPCS_a_o.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vec = np.concatenate((w1vecH_b_tr, w1vecPCS_b_tr))
            w2vec = np.concatenate((w2vecH_b_tr, w2vecPCS_b_tr))
            cossim = 1 - spatial.distance.cosine(w1vec, w2vec)
            cossim_listHPCS_b_tr.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vec = np.concatenate((w1vecH_b_o, w1vecPCS_b_o))
            w2vec = np.concatenate((w2vecH_b_o, w2vecPCS_b_o))
            cossim = 1 - spatial.distance.cosine(w1vec, w2vec)
            cossim_listHPCS_b_o.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            ldPhon = phonword.dl(word1PhonsPI, word2PhonsPI)
            ldNormPhon = phonword.dlNormalized(word1PhonsPI, word2PhonsPI)

            w1vec = np.concatenate((w1vecH_c_tr, w1vecPI_c_tr))
            w2vec = np.concatenate((w2vecH_c_tr, w2vecPI_c_tr))
            cossim = 1 - spatial.distance.cosine(w1vec, w2vec)
            cossim_listHPI_c_tr.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vec = np.concatenate((w1vecH_c_o, w1vecPI_c_o))
            w2vec = np.concatenate((w2vecH_c_o, w2vecPI_c_o))
            cossim = 1 - spatial.distance.cosine(w1vec, w2vec)
            cossim_listHPI_c_o.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vec = np.concatenate((w1vecH_a_tr, w1vecPI_a_tr))
            w2vec = np.concatenate((w2vecH_a_tr, w2vecPI_a_tr))
            cossim = 1 - spatial.distance.cosine(w1vec, w2vec)
            cossim_listHPI_a_tr.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vec = np.concatenate((w1vecH_a_o, w1vecPI_a_o))
            w2vec = np.concatenate((w2vecH_a_o, w2vecPI_a_o))
            cossim = 1 - spatial.distance.cosine(w1vec, w2vec)
            cossim_listHPI_a_o.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vec = np.concatenate((w1vecH_b_tr, w1vecPI_b_tr))
            w2vec = np.concatenate((w2vecH_b_tr, w2vecPI_b_tr))
            cossim = 1 - spatial.distance.cosine(w1vec, w2vec)
            cossim_listHPI_b_tr.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])

            w1vec = np.concatenate((w1vecH_b_o, w1vecPI_b_o))
            w2vec = np.concatenate((w2vecH_b_o, w2vecPI_b_o))
            cossim = 1 - spatial.distance.cosine(w1vec, w2vec)
            cossim_listHPI_b_o.append([abs(cossim), ld, ldNorm, ldPhon, ldNormPhon])
            


        


        #add lists of cosine similarities to df
        df['cossimH_c_tr'] = getCossims(cossim_listH_c_tr)
        df['cossimH_c_o'] = getCossims(cossim_listH_c_o)
        df['cossimH_a_tr'] = getCossims(cossim_listH_a_tr)
        df['cossimH_a_o'] = getCossims(cossim_listH_a_o)
        df['cossimH_b_tr'] = getCossims(cossim_listH_b_tr)
        df['cossimH_b_o'] = getCossims(cossim_listH_b_o)
        df['cossimPC_c_tr'] = getCossims(cossim_listPC_c_tr)
        df['cossimPC_c_o'] = getCossims(cossim_listPC_c_o)
        df['cossimPC_a_tr'] = getCossims(cossim_listPC_a_tr)
        df['cossimPC_a_o'] = getCossims(cossim_listPC_a_o)
        df['cossimPC_b_tr'] = getCossims(cossim_listPC_b_tr)
        df['cossimPC_b_o'] = getCossims(cossim_listPC_b_o)
        df['cossimPCS_c_tr'] = getCossims(cossim_listPCS_c_tr)
        df['cossimPCS_c_o'] = getCossims(cossim_listPCS_c_o)
        df['cossimPCS_a_tr'] = getCossims(cossim_listPCS_a_tr)
        df['cossimPCS_a_o'] = getCossims(cossim_listPCS_a_o)
        df['cossimPCS_b_tr'] = getCossims(cossim_listPCS_b_tr)
        df['cossimPCS_b_o'] = getCossims(cossim_listPCS_b_o)
        df['cossimPI_c_tr'] = getCossims(cossim_listPI_c_tr)
        df['cossimPI_c_o'] = getCossims(cossim_listPI_c_o)
        df['cossimPI_a_tr'] = getCossims(cossim_listPI_a_tr)
        df['cossimPI_a_o'] = getCossims(cossim_listPI_a_o)
        df['cossimPI_b_tr'] = getCossims(cossim_listPI_b_tr)
        df['cossimPI_b_o'] = getCossims(cossim_listPI_b_o)
        df['cossimHPC_c_tr'] = getCossims(cossim_listHPC_c_tr)
        df['cossimHPC_c_o'] = getCossims(cossim_listHPC_c_o)
        df['cossimHPC_a_tr'] = getCossims(cossim_listHPC_a_tr)
        df['cossimHPC_a_o'] = getCossims(cossim_listHPC_a_o)
        df['cossimHPC_b_tr'] = getCossims(cossim_listHPC_b_tr)
        df['cossimHPC_b_o'] = getCossims(cossim_listHPC_b_o)
        df['cossimHPCS_c_tr'] = getCossims(cossim_listHPCS_c_tr)
        df['cossimHPCS_c_o'] = getCossims(cossim_listHPCS_c_o)
        df['cossimHPCS_a_tr'] = getCossims(cossim_listHPCS_a_tr)
        df['cossimHPCS_a_o'] = getCossims(cossim_listHPCS_a_o)
        df['cossimHPCS_b_tr'] = getCossims(cossim_listHPCS_b_tr)
        df['cossimHPCS_b_o'] = getCossims(cossim_listHPCS_b_o)
        df['cossimHPI_c_tr'] = getCossims(cossim_listHPI_c_tr)
        df['cossimHPI_c_o'] = getCossims(cossim_listHPI_c_o)
        df['cossimHPI_a_tr'] = getCossims(cossim_listHPI_a_tr)
        df['cossimHPI_a_o'] = getCossims(cossim_listHPI_a_o)
        df['cossimHPI_b_tr'] = getCossims(cossim_listHPI_b_tr)
        df['cossimHPI_b_o'] = getCossims(cossim_listHPI_b_o)

        df['levDistH_c_tr'] = getlevDist(cossim_listH_c_tr)
        df['levDistH_c_o'] = getlevDist(cossim_listH_c_o)
        df['levDistH_a_tr'] = getlevDist(cossim_listH_a_tr)
        df['levDistH_a_o'] = getlevDist(cossim_listH_a_o)
        df['levDistH_b_tr'] = getlevDist(cossim_listH_b_tr)
        df['levDistH_b_o'] = getlevDist(cossim_listH_b_o)
        df['levDistPC_c_tr'] = getlevDist(cossim_listPC_c_tr)
        df['levDistPC_c_o'] = getlevDist(cossim_listPC_c_o)
        df['levDistPC_a_tr'] = getlevDist(cossim_listPC_a_tr)
        df['levDistPC_a_o'] = getlevDist(cossim_listPC_a_o)
        df['levDistPC_b_tr'] = getlevDist(cossim_listPC_b_tr)
        df['levDistPC_b_o'] = getlevDist(cossim_listPC_b_o)
        df['levDistPCS_c_tr'] = getlevDist(cossim_listPCS_c_tr)
        df['levDistPCS_c_o'] = getlevDist(cossim_listPCS_c_o)
        df['levDistPCS_a_tr'] = getlevDist(cossim_listPCS_a_tr)
        df['levDistPCS_a_o'] = getlevDist(cossim_listPCS_a_o)
        df['levDistPCS_b_tr'] = getlevDist(cossim_listPCS_b_tr)
        df['levDistPCS_b_o'] = getlevDist(cossim_listPCS_b_o)
        df['levDistPI_c_tr'] = getlevDist(cossim_listPI_c_tr)
        df['levDistPI_c_o'] = getlevDist(cossim_listPI_c_o)
        df['levDistPI_a_tr'] = getlevDist(cossim_listPI_a_tr)
        df['levDistPI_a_o'] = getlevDist(cossim_listPI_a_o)
        df['levDistPI_b_tr'] = getlevDist(cossim_listPI_b_tr)
        df['levDistPI_b_o'] = getlevDist(cossim_listPI_b_o)
        df['levDistHPC_c_tr'] = getlevDist(cossim_listHPC_c_tr)
        df['levDistHPC_c_o'] = getlevDist(cossim_listHPC_c_o)
        df['levDistHPC_a_tr'] = getlevDist(cossim_listHPC_a_tr)
        df['levDistHPC_a_o'] = getlevDist(cossim_listHPC_a_o)
        df['levDistHPC_b_tr'] = getlevDist(cossim_listHPC_b_tr)
        df['levDistHPC_b_o'] = getlevDist(cossim_listHPC_b_o)
        df['levDistHPCS_c_tr'] = getlevDist(cossim_listHPCS_c_tr)
        df['levDistHPCS_c_o'] = getlevDist(cossim_listHPCS_c_o)
        df['levDistHPCS_a_tr'] = getlevDist(cossim_listHPCS_a_tr)
        df['levDistHPCS_a_o'] = getlevDist(cossim_listHPCS_a_o)
        df['levDistHPCS_b_tr'] = getlevDist(cossim_listHPCS_b_tr)
        df['levDistHPCS_b_o'] = getlevDist(cossim_listHPCS_b_o)
        df['levDistHPI_c_tr'] = getlevDist(cossim_listHPI_c_tr)
        df['levDistHPI_c_o'] = getlevDist(cossim_listHPI_c_o)
        df['levDistHPI_a_tr'] = getlevDist(cossim_listHPI_a_tr)
        df['levDistHPI_a_o'] = getlevDist(cossim_listHPI_a_o)
        df['levDistHPI_b_tr'] = getlevDist(cossim_listHPI_b_tr)
        df['levDistHPI_b_o'] = getlevDist(cossim_listHPI_b_o)

        df['levDistNormH_c_tr'] = getlevDistNorm(cossim_listH_c_tr)
        df['levDistNormH_c_o'] = getlevDistNorm(cossim_listH_c_o)
        df['levDistNormH_a_tr'] = getlevDistNorm(cossim_listH_a_tr)
        df['levDistNormH_a_o'] = getlevDistNorm(cossim_listH_a_o)
        df['levDistNormH_b_tr'] = getlevDistNorm(cossim_listH_b_tr)
        df['levDistNormH_b_o'] = getlevDistNorm(cossim_listH_b_o)
        df['levDistNormPC_c_tr'] = getlevDistNorm(cossim_listPC_c_tr)
        df['levDistNormPC_c_o'] = getlevDistNorm(cossim_listPC_c_o)
        df['levDistNormPC_a_tr'] = getlevDistNorm(cossim_listPC_a_tr)
        df['levDistNormPC_a_o'] = getlevDistNorm(cossim_listPC_a_o)
        df['levDistNormPC_b_tr'] = getlevDistNorm(cossim_listPC_b_tr)
        df['levDistNormPC_b_o'] = getlevDistNorm(cossim_listPC_b_o)
        df['levDistNormPCS_c_tr'] = getlevDistNorm(cossim_listPCS_c_tr)
        df['levDistNormPCS_c_o'] = getlevDistNorm(cossim_listPCS_c_o)
        df['levDistNormPCS_a_tr'] = getlevDistNorm(cossim_listPCS_a_tr)
        df['levDistNormPCS_a_o'] = getlevDistNorm(cossim_listPCS_a_o)
        df['levDistNormPCS_b_tr'] = getlevDistNorm(cossim_listPCS_b_tr)
        df['levDistNormPCS_b_o'] = getlevDistNorm(cossim_listPCS_b_o)
        df['levDistNormPI_c_tr'] = getlevDistNorm(cossim_listPI_c_tr)
        df['levDistNormPI_c_o'] = getlevDistNorm(cossim_listPI_c_o)
        df['levDistNormPI_a_tr'] = getlevDistNorm(cossim_listPI_a_tr)
        df['levDistNormPI_a_o'] = getlevDistNorm(cossim_listPI_a_o)
        df['levDistNormPI_b_tr'] = getlevDistNorm(cossim_listPI_b_tr)
        df['levDistNormPI_b_o'] = getlevDistNorm(cossim_listPI_b_o)
        df['levDistNormHPC_c_tr'] = getlevDistNorm(cossim_listHPC_c_tr)
        df['levDistNormHPC_c_o'] = getlevDistNorm(cossim_listHPC_c_o)
        df['levDistNormHPC_a_tr'] = getlevDistNorm(cossim_listHPC_a_tr)
        df['levDistNormHPC_a_o'] = getlevDistNorm(cossim_listHPC_a_o)
        df['levDistNormHPC_b_tr'] = getlevDistNorm(cossim_listHPC_b_tr)
        df['levDistNormHPC_b_o'] = getlevDistNorm(cossim_listHPC_b_o)
        df['levDistNormHPCS_c_tr'] = getlevDistNorm(cossim_listHPCS_c_tr)
        df['levDistNormHPCS_c_o'] = getlevDistNorm(cossim_listHPCS_c_o)
        df['levDistNormHPCS_a_tr'] = getlevDistNorm(cossim_listHPCS_a_tr)
        df['levDistNormHPCS_a_o'] = getlevDistNorm(cossim_listHPCS_a_o)
        df['levDistNormHPCS_b_tr'] = getlevDistNorm(cossim_listHPCS_b_tr)
        df['levDistNormHPCS_b_o'] = getlevDistNorm(cossim_listHPCS_b_o)
        df['levDistNormHPI_c_tr'] = getlevDistNorm(cossim_listHPI_c_tr)
        df['levDistNormHPI_c_o'] = getlevDistNorm(cossim_listHPI_c_o)
        df['levDistNormHPI_a_tr'] = getlevDistNorm(cossim_listHPI_a_tr)
        df['levDistNormHPI_a_o'] = getlevDistNorm(cossim_listHPI_a_o)
        df['levDistNormHPI_b_tr'] = getlevDistNorm(cossim_listHPI_b_tr)
        df['levDistNormHPI_b_o'] = getlevDistNorm(cossim_listHPI_b_o)

        df['levDistPhonH_c_tr'] = getlevDistPhon(cossim_listH_c_tr)
        df['levDistPhonH_c_o'] = getlevDistPhon(cossim_listH_c_o)
        df['levDistPhonH_a_tr'] = getlevDistPhon(cossim_listH_a_tr)
        df['levDistPhonH_a_o'] = getlevDistPhon(cossim_listH_a_o)
        df['levDistPhonH_b_tr'] = getlevDistPhon(cossim_listH_b_tr)
        df['levDistPhonH_b_o'] = getlevDistPhon(cossim_listH_b_o)
        df['levDistPhonPC_c_tr'] = getlevDistPhon(cossim_listPC_c_tr)
        df['levDistPhonPC_c_o'] = getlevDistPhon(cossim_listPC_c_o)
        df['levDistPhonPC_a_tr'] = getlevDistPhon(cossim_listPC_a_tr)
        df['levDistPhonPC_a_o'] = getlevDistPhon(cossim_listPC_a_o)
        df['levDistPhonPC_b_tr'] = getlevDistPhon(cossim_listPC_b_tr)
        df['levDistPhonPC_b_o'] = getlevDistPhon(cossim_listPC_b_o)
        df['levDistPhonPCS_c_tr'] = getlevDistPhon(cossim_listPCS_c_tr)
        df['levDistPhonPCS_c_o'] = getlevDistPhon(cossim_listPCS_c_o)
        df['levDistPhonPCS_a_tr'] = getlevDistPhon(cossim_listPCS_a_tr)
        df['levDistPhonPCS_a_o'] = getlevDistPhon(cossim_listPCS_a_o)
        df['levDistPhonPCS_b_tr'] = getlevDistPhon(cossim_listPCS_b_tr)
        df['levDistPhonPCS_b_o'] = getlevDistPhon(cossim_listPCS_b_o)
        df['levDistPhonPI_c_tr'] = getlevDistPhon(cossim_listPI_c_tr)
        df['levDistPhonPI_c_o'] = getlevDistPhon(cossim_listPI_c_o)
        df['levDistPhonPI_a_tr'] = getlevDistPhon(cossim_listPI_a_tr)
        df['levDistPhonPI_a_o'] = getlevDistPhon(cossim_listPI_a_o)
        df['levDistPhonPI_b_tr'] = getlevDistPhon(cossim_listPI_b_tr)
        df['levDistPhonPI_b_o'] = getlevDistPhon(cossim_listPI_b_o)
        df['levDistPhonHPC_c_tr'] = getlevDistPhon(cossim_listHPC_c_tr)
        df['levDistPhonHPC_c_o'] = getlevDistPhon(cossim_listHPC_c_o)
        df['levDistPhonHPC_a_tr'] = getlevDistPhon(cossim_listHPC_a_tr)
        df['levDistPhonHPC_a_o'] = getlevDistPhon(cossim_listHPC_a_o)
        df['levDistPhonHPC_b_tr'] = getlevDistPhon(cossim_listHPC_b_tr)
        df['levDistPhonHPC_b_o'] = getlevDistPhon(cossim_listHPC_b_o)
        df['levDistPhonHPCS_c_tr'] = getlevDistPhon(cossim_listHPCS_c_tr)
        df['levDistPhonHPCS_c_o'] = getlevDistPhon(cossim_listHPCS_c_o)
        df['levDistPhonHPCS_a_tr'] = getlevDistPhon(cossim_listHPCS_a_tr)
        df['levDistPhonHPCS_a_o'] = getlevDistPhon(cossim_listHPCS_a_o)
        df['levDistPhonHPCS_b_tr'] = getlevDistPhon(cossim_listHPCS_b_tr)
        df['levDistPhonHPCS_b_o'] = getlevDistPhon(cossim_listHPCS_b_o)
        df['levDistPhonHPI_c_tr'] = getlevDistPhon(cossim_listHPI_c_tr)
        df['levDistPhonHPI_c_o'] = getlevDistPhon(cossim_listHPI_c_o)
        df['levDistPhonHPI_a_tr'] = getlevDistPhon(cossim_listHPI_a_tr)
        df['levDistPhonHPI_a_o'] = getlevDistPhon(cossim_listHPI_a_o)
        df['levDistPhonHPI_b_tr'] = getlevDistPhon(cossim_listHPI_b_tr)
        df['levDistPhonHPI_b_o'] = getlevDistPhon(cossim_listHPI_b_o)

        df['levDistNormPhonH_c_tr'] = getlevDistNormPhon(cossim_listH_c_tr)
        df['levDistNormPhonH_c_o'] = getlevDistNormPhon(cossim_listH_c_o)
        df['levDistNormPhonH_a_tr'] = getlevDistNormPhon(cossim_listH_a_tr)
        df['levDistNormPhonH_a_o'] = getlevDistNormPhon(cossim_listH_a_o)
        df['levDistNormPhonH_b_tr'] = getlevDistNormPhon(cossim_listH_b_tr)
        df['levDistNormPhonH_b_o'] = getlevDistNormPhon(cossim_listH_b_o)
        df['levDistNormPhonPC_c_tr'] = getlevDistNormPhon(cossim_listPC_c_tr)
        df['levDistNormPhonPC_c_o'] = getlevDistNormPhon(cossim_listPC_c_o)
        df['levDistNormPhonPC_a_tr'] = getlevDistNormPhon(cossim_listPC_a_tr)
        df['levDistNormPhonPC_a_o'] = getlevDistNormPhon(cossim_listPC_a_o)
        df['levDistNormPhonPC_b_tr'] = getlevDistNormPhon(cossim_listPC_b_tr)
        df['levDistNormPhonPC_b_o'] = getlevDistNormPhon(cossim_listPC_b_o)
        df['levDistNormPhonPCS_c_tr'] = getlevDistNormPhon(cossim_listPCS_c_tr)
        df['levDistNormPhonPCS_c_o'] = getlevDistNormPhon(cossim_listPCS_c_o)
        df['levDistNormPhonPCS_a_tr'] = getlevDistNormPhon(cossim_listPCS_a_tr)
        df['levDistNormPhonPCS_a_o'] = getlevDistNormPhon(cossim_listPCS_a_o)
        df['levDistNormPhonPCS_b_tr'] = getlevDistNormPhon(cossim_listPCS_b_tr)
        df['levDistNormPhonPCS_b_o'] = getlevDistNormPhon(cossim_listPCS_b_o)
        df['levDistNormPhonPI_c_tr'] = getlevDistNormPhon(cossim_listPI_c_tr)
        df['levDistNormPhonPI_c_o'] = getlevDistNormPhon(cossim_listPI_c_o)
        df['levDistNormPhonPI_a_tr'] = getlevDistNormPhon(cossim_listPI_a_tr)
        df['levDistNormPhonPI_a_o'] = getlevDistNormPhon(cossim_listPI_a_o)
        df['levDistNormPhonPI_b_tr'] = getlevDistNormPhon(cossim_listPI_b_tr)
        df['levDistNormPhonPI_b_o'] = getlevDistNormPhon(cossim_listPI_b_o)
        df['levDistNormPhonHPC_c_tr'] = getlevDistNormPhon(cossim_listHPC_c_tr)
        df['levDistNormPhonHPC_c_o'] = getlevDistNormPhon(cossim_listHPC_c_o)
        df['levDistNormPhonHPC_a_tr'] = getlevDistNormPhon(cossim_listHPC_a_tr)
        df['levDistNormPhonHPC_a_o'] = getlevDistNormPhon(cossim_listHPC_a_o)
        df['levDistNormPhonHPC_b_tr'] = getlevDistNormPhon(cossim_listHPC_b_tr)
        df['levDistNormPhonHPC_b_o'] = getlevDistNormPhon(cossim_listHPC_b_o)
        df['levDistNormPhonHPCS_c_tr'] = getlevDistNormPhon(cossim_listHPCS_c_tr)
        df['levDistNormPhonHPCS_c_o'] = getlevDistNormPhon(cossim_listHPCS_c_o)
        df['levDistNormPhonHPCS_a_tr'] = getlevDistNormPhon(cossim_listHPCS_a_tr)
        df['levDistNormPhonHPCS_a_o'] = getlevDistNormPhon(cossim_listHPCS_a_o)
        df['levDistNormPhonHPCS_b_tr'] = getlevDistNormPhon(cossim_listHPCS_b_tr)
        df['levDistNormPhonHPCS_b_o'] = getlevDistNormPhon(cossim_listHPCS_b_o)
        df['levDistNormPhonHPI_c_tr'] = getlevDistNormPhon(cossim_listHPI_c_tr)
        df['levDistNormPhonHPI_c_o'] = getlevDistNormPhon(cossim_listHPI_c_o)
        df['levDistNormPhonHPI_a_tr'] = getlevDistNormPhon(cossim_listHPI_a_tr)
        df['levDistNormPhonHPI_a_o'] = getlevDistNormPhon(cossim_listHPI_a_o)
        df['levDistNormPhonHPI_b_tr'] = getlevDistNormPhon(cossim_listHPI_b_tr)
        df['levDistNormPhonHPI_b_o'] = getlevDistNormPhon(cossim_listHPI_b_o)

        #add correlation between each cosine similarity list and testing parameter to each dictionary
        H_c_tr['simRating'].append(df["Response"].corr(df["cossimH_c_tr"]))
        H_c_tr['phonDistance'].append(df["distance_x"].corr(df["cossimH_c_tr"]))
        H_c_tr['levDistanceOrtho'].append(df["levDistH_c_tr"].corr(df["cossimH_c_tr"]))
        H_c_tr['levDistanceNormalizedOrtho'].append(df["levDistNormH_c_tr"].corr(df["cossimH_c_tr"]))
        H_c_tr['levDistancePhon'].append(df["levDistPhonH_c_tr"].corr(df["cossimH_c_tr"]))
        H_c_tr['levDistanceNormalizedPhon'].append(df["levDistNormPhonH_c_tr"].corr(df["cossimH_c_tr"]))

        H_c_o['simRating'].append(df["Response"].corr(df["cossimH_c_o"]))
        H_c_o['phonDistance'].append(df["distance_x"].corr(df["cossimH_c_o"]))
        H_c_o['levDistanceOrtho'].append(df["levDistH_c_o"].corr(df["cossimH_c_o"]))
        H_c_o['levDistanceNormalizedOrtho'].append(df["levDistNormH_c_o"].corr(df["cossimH_c_o"]))
        H_c_o['levDistancePhon'].append(df["levDistPhonH_c_o"].corr(df["cossimH_c_o"]))
        H_c_o['levDistanceNormalizedPhon'].append(df["levDistNormPhonH_c_o"].corr(df["cossimH_c_o"]))

        H_a_tr['simRating'].append(df["Response"].corr(df["cossimH_a_tr"]))
        H_a_tr['phonDistance'].append(df["distance_x"].corr(df["cossimH_a_tr"]))
        H_a_tr['levDistanceOrtho'].append(df["levDistH_a_tr"].corr(df["cossimH_a_tr"]))
        H_a_tr['levDistanceNormalizedOrtho'].append(df["levDistNormH_a_tr"].corr(df["cossimH_a_tr"]))
        H_a_tr['levDistancePhon'].append(df["levDistPhonH_a_tr"].corr(df["cossimH_a_tr"]))
        H_a_tr['levDistanceNormalizedPhon'].append(df["levDistNormPhonH_a_tr"].corr(df["cossimH_a_tr"]))

        H_a_o['simRating'].append(df["Response"].corr(df["cossimH_a_o"]))
        H_a_o['phonDistance'].append(df["distance_x"].corr(df["cossimH_a_o"]))
        H_a_o['levDistanceOrtho'].append(df["levDistH_a_o"].corr(df["cossimH_a_o"]))
        H_a_o['levDistanceNormalizedOrtho'].append(df["levDistNormH_a_o"].corr(df["cossimH_a_o"]))
        H_a_o['levDistancePhon'].append(df["levDistPhonH_a_o"].corr(df["cossimH_a_o"]))
        H_a_o['levDistanceNormalizedPhon'].append(df["levDistNormPhonH_a_o"].corr(df["cossimH_a_o"]))

        H_b_tr['simRating'].append(df["Response"].corr(df["cossimH_b_tr"]))
        H_b_tr['phonDistance'].append(df["distance_x"].corr(df["cossimH_b_tr"]))
        H_b_tr['levDistanceOrtho'].append(df["levDistH_b_tr"].corr(df["cossimH_b_tr"]))
        H_b_tr['levDistanceNormalizedOrtho'].append(df["levDistNormH_b_tr"].corr(df["cossimH_b_tr"]))
        H_b_tr['levDistancePhon'].append(df["levDistPhonH_b_tr"].corr(df["cossimH_b_tr"]))
        H_b_tr['levDistanceNormalizedPhon'].append(df["levDistNormPhonH_b_tr"].corr(df["cossimH_b_tr"]))

        H_b_o['simRating'].append(df["Response"].corr(df["cossimH_b_o"]))
        H_b_o['phonDistance'].append(df["distance_x"].corr(df["cossimH_b_o"]))
        H_b_o['levDistanceOrtho'].append(df["levDistH_b_o"].corr(df["cossimH_b_o"]))
        H_b_o['levDistanceNormalizedOrtho'].append(df["levDistNormH_b_o"].corr(df["cossimH_b_o"]))
        H_b_o['levDistancePhon'].append(df["levDistPhonH_b_o"].corr(df["cossimH_b_o"]))
        H_b_o['levDistanceNormalizedPhon'].append(df["levDistNormPhonH_b_o"].corr(df["cossimH_b_o"]))

        PC_c_tr['simRating'].append(df["Response"].corr(df["cossimPC_c_tr"]))
        PC_c_tr['phonDistance'].append(df["distance_x"].corr(df["cossimPC_c_tr"]))
        PC_c_tr['levDistanceOrtho'].append(df["levDistPC_c_tr"].corr(df["cossimPC_c_tr"]))
        PC_c_tr['levDistanceNormalizedOrtho'].append(df["levDistNormPC_c_tr"].corr(df["cossimPC_c_tr"]))
        PC_c_tr['levDistancePhon'].append(df["levDistPhonPC_c_tr"].corr(df["cossimPC_c_tr"]))
        PC_c_tr['levDistanceNormalizedPhon'].append(df["levDistNormPhonPC_c_tr"].corr(df["cossimPC_c_tr"]))

        PC_c_o['simRating'].append(df["Response"].corr(df["cossimPC_c_o"]))
        PC_c_o['phonDistance'].append(df["distance_x"].corr(df["cossimPC_c_o"]))
        PC_c_o['levDistanceOrtho'].append(df["levDistPC_c_o"].corr(df["cossimPC_c_o"]))
        PC_c_o['levDistanceNormalizedOrtho'].append(df["levDistNormPC_c_o"].corr(df["cossimPC_c_o"]))
        PC_c_o['levDistancePhon'].append(df["levDistPhonPC_c_o"].corr(df["cossimPC_c_o"]))
        PC_c_o['levDistanceNormalizedPhon'].append(df["levDistNormPhonPC_c_o"].corr(df["cossimPC_c_o"]))

        PC_a_tr['simRating'].append(df["Response"].corr(df["cossimPC_a_tr"]))
        PC_a_tr['phonDistance'].append(df["distance_x"].corr(df["cossimPC_a_tr"]))
        PC_a_tr['levDistanceOrtho'].append(df["levDistPC_a_tr"].corr(df["cossimPC_a_tr"]))
        PC_a_tr['levDistanceNormalizedOrtho'].append(df["levDistNormPC_a_tr"].corr(df["cossimPC_a_tr"]))
        PC_a_tr['levDistancePhon'].append(df["levDistPhonPC_a_tr"].corr(df["cossimPC_a_tr"]))
        PC_a_tr['levDistanceNormalizedPhon'].append(df["levDistNormPhonPC_a_tr"].corr(df["cossimPC_a_tr"]))

        PC_a_o['simRating'].append(df["Response"].corr(df["cossimPC_a_o"]))
        PC_a_o['phonDistance'].append(df["distance_x"].corr(df["cossimPC_a_o"]))
        PC_a_o['levDistanceOrtho'].append(df["levDistPC_a_o"].corr(df["cossimPC_a_o"]))
        PC_a_o['levDistanceNormalizedOrtho'].append(df["levDistNormPC_a_o"].corr(df["cossimPC_a_o"]))
        PC_a_o['levDistancePhon'].append(df["levDistPhonPC_a_o"].corr(df["cossimPC_a_o"]))
        PC_a_o['levDistanceNormalizedPhon'].append(df["levDistNormPhonPC_a_o"].corr(df["cossimPC_a_o"]))

        PC_b_tr['simRating'].append(df["Response"].corr(df["cossimPC_b_tr"]))
        PC_b_tr['phonDistance'].append(df["distance_x"].corr(df["cossimPC_b_tr"]))
        PC_b_tr['levDistanceOrtho'].append(df["levDistPC_b_tr"].corr(df["cossimPC_b_tr"]))
        PC_b_tr['levDistanceNormalizedOrtho'].append(df["levDistNormPC_b_tr"].corr(df["cossimPC_b_tr"]))
        PC_b_tr['levDistancePhon'].append(df["levDistPhonPC_b_tr"].corr(df["cossimPC_b_tr"]))
        PC_b_tr['levDistanceNormalizedPhon'].append(df["levDistNormPhonPC_b_tr"].corr(df["cossimPC_b_tr"]))

        PC_b_o['simRating'].append(df["Response"].corr(df["cossimPC_b_o"]))
        PC_b_o['phonDistance'].append(df["distance_x"].corr(df["cossimPC_b_o"]))
        PC_b_o['levDistanceOrtho'].append(df["levDistPC_b_o"].corr(df["cossimPC_b_o"]))
        PC_b_o['levDistanceNormalizedOrtho'].append(df["levDistNormPC_b_o"].corr(df["cossimPC_b_o"]))
        PC_b_o['levDistancePhon'].append(df["levDistPhonPC_b_o"].corr(df["cossimPC_b_o"]))
        PC_b_o['levDistanceNormalizedPhon'].append(df["levDistNormPhonPC_b_o"].corr(df["cossimPC_b_o"]))

        PCS_c_tr['simRating'].append(df["Response"].corr(df["cossimPCS_c_tr"]))
        PCS_c_tr['phonDistance'].append(df["distance_x"].corr(df["cossimPCS_c_tr"]))
        PCS_c_tr['levDistanceOrtho'].append(df["levDistPCS_c_tr"].corr(df["cossimPCS_c_tr"]))
        PCS_c_tr['levDistanceNormalizedOrtho'].append(df["levDistNormPCS_c_tr"].corr(df["cossimPCS_c_tr"]))
        PCS_c_tr['levDistancePhon'].append(df["levDistPhonPCS_c_tr"].corr(df["cossimPCS_c_tr"]))
        PCS_c_tr['levDistanceNormalizedPhon'].append(df["levDistNormPhonPCS_c_tr"].corr(df["cossimPCS_c_tr"]))

        PCS_c_o['simRating'].append(df["Response"].corr(df["cossimPCS_c_o"]))
        PCS_c_o['phonDistance'].append(df["distance_x"].corr(df["cossimPCS_c_o"]))
        PCS_c_o['levDistanceOrtho'].append(df["levDistPCS_c_o"].corr(df["cossimPCS_c_o"]))
        PCS_c_o['levDistanceNormalizedOrtho'].append(df["levDistNormPCS_c_o"].corr(df["cossimPCS_c_o"]))
        PCS_c_o['levDistancePhon'].append(df["levDistPhonPCS_c_o"].corr(df["cossimPCS_c_o"]))
        PCS_c_o['levDistanceNormalizedPhon'].append(df["levDistNormPhonPCS_c_o"].corr(df["cossimPCS_c_o"]))

        PCS_a_tr['simRating'].append(df["Response"].corr(df["cossimPCS_a_tr"]))
        PCS_a_tr['phonDistance'].append(df["distance_x"].corr(df["cossimPCS_a_tr"]))
        PCS_a_tr['levDistanceOrtho'].append(df["levDistPCS_a_tr"].corr(df["cossimPCS_a_tr"]))
        PCS_a_tr['levDistanceNormalizedOrtho'].append(df["levDistNormPCS_a_tr"].corr(df["cossimPCS_a_tr"]))
        PCS_a_tr['levDistancePhon'].append(df["levDistPhonPCS_a_tr"].corr(df["cossimPCS_a_tr"]))
        PCS_a_tr['levDistanceNormalizedPhon'].append(df["levDistNormPhonPCS_a_tr"].corr(df["cossimPCS_a_tr"]))

        PCS_a_o['simRating'].append(df["Response"].corr(df["cossimPCS_a_o"]))
        PCS_a_o['phonDistance'].append(df["distance_x"].corr(df["cossimPCS_a_o"]))
        PCS_a_o['levDistanceOrtho'].append(df["levDistPCS_a_o"].corr(df["cossimPCS_a_o"]))
        PCS_a_o['levDistanceNormalizedOrtho'].append(df["levDistNormPCS_a_o"].corr(df["cossimPCS_a_o"]))
        PCS_a_o['levDistancePhon'].append(df["levDistPhonPCS_a_o"].corr(df["cossimPCS_a_o"]))
        PCS_a_o['levDistanceNormalizedPhon'].append(df["levDistNormPhonPCS_a_o"].corr(df["cossimPCS_a_o"]))

        PCS_b_tr['simRating'].append(df["Response"].corr(df["cossimPCS_b_tr"]))
        PCS_b_tr['phonDistance'].append(df["distance_x"].corr(df["cossimPCS_b_tr"]))
        PCS_b_tr['levDistanceOrtho'].append(df["levDistPCS_b_tr"].corr(df["cossimPCS_b_tr"]))
        PCS_b_tr['levDistanceNormalizedOrtho'].append(df["levDistNormPCS_b_tr"].corr(df["cossimPCS_b_tr"]))
        PCS_b_tr['levDistancePhon'].append(df["levDistPhonPCS_b_tr"].corr(df["cossimPCS_b_tr"]))
        PCS_b_tr['levDistanceNormalizedPhon'].append(df["levDistNormPhonPCS_b_tr"].corr(df["cossimPCS_b_tr"]))

        PCS_b_o['simRating'].append(df["Response"].corr(df["cossimPCS_b_o"]))
        PCS_b_o['phonDistance'].append(df["distance_x"].corr(df["cossimPCS_b_o"]))
        PCS_b_o['levDistanceOrtho'].append(df["levDistPCS_b_o"].corr(df["cossimPCS_b_o"]))
        PCS_b_o['levDistanceNormalizedOrtho'].append(df["levDistNormPCS_b_o"].corr(df["cossimPCS_b_o"]))
        PCS_b_o['levDistancePhon'].append(df["levDistPhonPCS_b_o"].corr(df["cossimPCS_b_o"]))
        PCS_b_o['levDistanceNormalizedPhon'].append(df["levDistNormPhonPCS_b_o"].corr(df["cossimPCS_b_o"]))

        PI_c_tr['simRating'].append(df["Response"].corr(df["cossimPI_c_tr"]))
        PI_c_tr['phonDistance'].append(df["distance_x"].corr(df["cossimPI_c_tr"]))
        PI_c_tr['levDistanceOrtho'].append(df["levDistPI_c_tr"].corr(df["cossimPI_c_tr"]))
        PI_c_tr['levDistanceNormalizedOrtho'].append(df["levDistNormPI_c_tr"].corr(df["cossimPI_c_tr"]))
        PI_c_tr['levDistancePhon'].append(df["levDistPhonPI_c_tr"].corr(df["cossimPI_c_tr"]))
        PI_c_tr['levDistanceNormalizedPhon'].append(df["levDistNormPhonPI_c_tr"].corr(df["cossimPI_c_tr"]))

        PI_c_o['simRating'].append(df["Response"].corr(df["cossimPI_c_o"]))
        PI_c_o['phonDistance'].append(df["distance_x"].corr(df["cossimPI_c_o"]))
        PI_c_o['levDistanceOrtho'].append(df["levDistPI_c_o"].corr(df["cossimPI_c_o"]))
        PI_c_o['levDistanceNormalizedOrtho'].append(df["levDistNormPI_c_o"].corr(df["cossimPI_c_o"]))
        PI_c_o['levDistancePhon'].append(df["levDistPhonPI_c_o"].corr(df["cossimPI_c_o"]))
        PI_c_o['levDistanceNormalizedPhon'].append(df["levDistNormPhonPI_c_o"].corr(df["cossimPI_c_o"]))

        PI_a_tr['simRating'].append(df["Response"].corr(df["cossimPI_a_tr"]))
        PI_a_tr['phonDistance'].append(df["distance_x"].corr(df["cossimPI_a_tr"]))
        PI_a_tr['levDistanceOrtho'].append(df["levDistPI_a_tr"].corr(df["cossimPI_a_tr"]))
        PI_a_tr['levDistanceNormalizedOrtho'].append(df["levDistNormPI_a_tr"].corr(df["cossimPI_a_tr"]))
        PI_a_tr['levDistancePhon'].append(df["levDistPhonPI_a_tr"].corr(df["cossimPI_a_tr"]))
        PI_a_tr['levDistanceNormalizedPhon'].append(df["levDistNormPhonPI_a_tr"].corr(df["cossimPI_a_tr"]))

        PI_a_o['simRating'].append(df["Response"].corr(df["cossimPI_a_o"]))
        PI_a_o['phonDistance'].append(df["distance_x"].corr(df["cossimPI_a_o"]))
        PI_a_o['levDistanceOrtho'].append(df["levDistPI_a_o"].corr(df["cossimPI_a_o"]))
        PI_a_o['levDistanceNormalizedOrtho'].append(df["levDistNormPI_a_o"].corr(df["cossimPI_a_o"]))
        PI_a_o['levDistancePhon'].append(df["levDistPhonPI_a_o"].corr(df["cossimPI_a_o"]))
        PI_a_o['levDistanceNormalizedPhon'].append(df["levDistNormPhonPI_a_o"].corr(df["cossimPI_a_o"]))

        PI_b_tr['simRating'].append(df["Response"].corr(df["cossimPI_b_tr"]))
        PI_b_tr['phonDistance'].append(df["distance_x"].corr(df["cossimPI_b_tr"]))
        PI_b_tr['levDistanceOrtho'].append(df["levDistPI_b_tr"].corr(df["cossimPI_b_tr"]))
        PI_b_tr['levDistanceNormalizedOrtho'].append(df["levDistNormPI_b_tr"].corr(df["cossimPI_b_tr"]))
        PI_b_tr['levDistancePhon'].append(df["levDistPhonPI_b_tr"].corr(df["cossimPI_b_tr"]))
        PI_b_tr['levDistanceNormalizedPhon'].append(df["levDistNormPhonPI_b_tr"].corr(df["cossimPI_b_tr"]))

        PI_b_o['simRating'].append(df["Response"].corr(df["cossimPI_b_o"]))
        PI_b_o['phonDistance'].append(df["distance_x"].corr(df["cossimPI_b_o"]))
        PI_b_o['levDistanceOrtho'].append(df["levDistPI_b_o"].corr(df["cossimPI_b_o"]))
        PI_b_o['levDistanceNormalizedOrtho'].append(df["levDistNormPI_b_o"].corr(df["cossimPI_b_o"]))
        PI_b_o['levDistancePhon'].append(df["levDistPhonPI_b_o"].corr(df["cossimPI_b_o"]))
        PI_b_o['levDistanceNormalizedPhon'].append(df["levDistNormPhonPI_b_o"].corr(df["cossimPI_b_o"]))

        HPC_c_tr['simRating'].append(df["Response"].corr(df["cossimHPC_c_tr"]))
        HPC_c_tr['phonDistance'].append(df["distance_x"].corr(df["cossimHPC_c_tr"]))
        HPC_c_tr['levDistanceOrtho'].append(df["levDistHPC_c_tr"].corr(df["cossimHPC_c_tr"]))
        HPC_c_tr['levDistanceNormalizedOrtho'].append(df["levDistNormHPC_c_tr"].corr(df["cossimHPC_c_tr"]))
        HPC_c_tr['levDistancePhon'].append(df["levDistPhonHPC_c_tr"].corr(df["cossimHPC_c_tr"]))
        HPC_c_tr['levDistanceNormalizedPhon'].append(df["levDistNormPhonHPC_c_tr"].corr(df["cossimHPC_c_tr"]))

        HPC_c_o['simRating'].append(df["Response"].corr(df["cossimHPC_c_o"]))
        HPC_c_o['phonDistance'].append(df["distance_x"].corr(df["cossimHPC_c_o"]))
        HPC_c_o['levDistanceOrtho'].append(df["levDistHPC_c_o"].corr(df["cossimHPC_c_o"]))
        HPC_c_o['levDistanceNormalizedOrtho'].append(df["levDistNormHPC_c_o"].corr(df["cossimHPC_c_o"]))
        HPC_c_o['levDistancePhon'].append(df["levDistPhonHPC_c_o"].corr(df["cossimHPC_c_o"]))
        HPC_c_o['levDistanceNormalizedPhon'].append(df["levDistNormPhonHPC_c_o"].corr(df["cossimHPC_c_o"]))

        HPC_a_tr['simRating'].append(df["Response"].corr(df["cossimHPC_a_tr"]))
        HPC_a_tr['phonDistance'].append(df["distance_x"].corr(df["cossimHPC_a_tr"]))
        HPC_a_tr['levDistanceOrtho'].append(df["levDistHPC_a_tr"].corr(df["cossimHPC_a_tr"]))
        HPC_a_tr['levDistanceNormalizedOrtho'].append(df["levDistNormHPC_a_tr"].corr(df["cossimHPC_a_tr"]))
        HPC_a_tr['levDistancePhon'].append(df["levDistPhonHPC_a_tr"].corr(df["cossimHPC_a_tr"]))
        HPC_a_tr['levDistanceNormalizedPhon'].append(df["levDistNormPhonHPC_a_tr"].corr(df["cossimHPC_a_tr"]))

        HPC_a_o['simRating'].append(df["Response"].corr(df["cossimHPC_a_o"]))
        HPC_a_o['phonDistance'].append(df["distance_x"].corr(df["cossimHPC_a_o"]))
        HPC_a_o['levDistanceOrtho'].append(df["levDistHPC_a_o"].corr(df["cossimHPC_a_o"]))
        HPC_a_o['levDistanceNormalizedOrtho'].append(df["levDistNormHPC_a_o"].corr(df["cossimHPC_a_o"]))
        HPC_a_o['levDistancePhon'].append(df["levDistPhonHPC_a_o"].corr(df["cossimHPC_a_o"]))
        HPC_a_o['levDistanceNormalizedPhon'].append(df["levDistNormPhonHPC_a_o"].corr(df["cossimHPC_a_o"]))

        HPC_b_tr['simRating'].append(df["Response"].corr(df["cossimHPC_b_tr"]))
        HPC_b_tr['phonDistance'].append(df["distance_x"].corr(df["cossimHPC_b_tr"]))
        HPC_b_tr['levDistanceOrtho'].append(df["levDistHPC_b_tr"].corr(df["cossimHPC_b_tr"]))
        HPC_b_tr['levDistanceNormalizedOrtho'].append(df["levDistNormHPC_b_tr"].corr(df["cossimHPC_b_tr"]))
        HPC_b_tr['levDistancePhon'].append(df["levDistPhonHPC_b_tr"].corr(df["cossimHPC_b_tr"]))
        HPC_b_tr['levDistanceNormalizedPhon'].append(df["levDistNormPhonHPC_b_tr"].corr(df["cossimHPC_b_tr"]))

        HPC_b_o['simRating'].append(df["Response"].corr(df["cossimHPC_b_o"]))
        HPC_b_o['phonDistance'].append(df["distance_x"].corr(df["cossimHPC_b_o"]))
        HPC_b_o['levDistanceOrtho'].append(df["levDistHPC_b_o"].corr(df["cossimHPC_b_o"]))
        HPC_b_o['levDistanceNormalizedOrtho'].append(df["levDistNormHPC_b_o"].corr(df["cossimHPC_b_o"]))
        HPC_b_o['levDistancePhon'].append(df["levDistPhonHPC_b_o"].corr(df["cossimHPC_b_o"]))
        HPC_b_o['levDistanceNormalizedPhon'].append(df["levDistNormPhonHPC_b_o"].corr(df["cossimHPC_b_o"]))

        HPCS_c_tr['simRating'].append(df["Response"].corr(df["cossimHPCS_c_tr"]))
        HPCS_c_tr['phonDistance'].append(df["distance_x"].corr(df["cossimHPCS_c_tr"]))
        HPCS_c_tr['levDistanceOrtho'].append(df["levDistHPCS_c_tr"].corr(df["cossimHPCS_c_tr"]))
        HPCS_c_tr['levDistanceNormalizedOrtho'].append(df["levDistNormHPCS_c_tr"].corr(df["cossimHPCS_c_tr"]))
        HPCS_c_tr['levDistancePhon'].append(df["levDistPhonHPCS_c_tr"].corr(df["cossimHPCS_c_tr"]))
        HPCS_c_tr['levDistanceNormalizedPhon'].append(df["levDistNormPhonHPCS_c_tr"].corr(df["cossimHPCS_c_tr"]))

        HPCS_c_o['simRating'].append(df["Response"].corr(df["cossimHPCS_c_o"]))
        HPCS_c_o['phonDistance'].append(df["distance_x"].corr(df["cossimHPCS_c_o"]))
        HPCS_c_o['levDistanceOrtho'].append(df["levDistHPCS_c_o"].corr(df["cossimHPCS_c_o"]))
        HPCS_c_o['levDistanceNormalizedOrtho'].append(df["levDistNormHPCS_c_o"].corr(df["cossimHPCS_c_o"]))
        HPCS_c_o['levDistancePhon'].append(df["levDistPhonHPCS_c_o"].corr(df["cossimHPCS_c_o"]))
        HPCS_c_o['levDistanceNormalizedPhon'].append(df["levDistNormPhonHPCS_c_o"].corr(df["cossimHPCS_c_o"]))

        HPCS_a_tr['simRating'].append(df["Response"].corr(df["cossimHPCS_a_tr"]))
        HPCS_a_tr['phonDistance'].append(df["distance_x"].corr(df["cossimHPCS_a_tr"]))
        HPCS_a_tr['levDistanceOrtho'].append(df["levDistHPCS_a_tr"].corr(df["cossimHPCS_a_tr"]))
        HPCS_a_tr['levDistanceNormalizedOrtho'].append(df["levDistNormHPCS_a_tr"].corr(df["cossimHPCS_a_tr"]))
        HPCS_a_tr['levDistancePhon'].append(df["levDistPhonHPCS_a_tr"].corr(df["cossimHPCS_a_tr"]))
        HPCS_a_tr['levDistanceNormalizedPhon'].append(df["levDistNormPhonHPCS_a_tr"].corr(df["cossimHPCS_a_tr"]))

        HPCS_a_o['simRating'].append(df["Response"].corr(df["cossimHPCS_a_o"]))
        HPCS_a_o['phonDistance'].append(df["distance_x"].corr(df["cossimHPCS_a_o"]))
        HPCS_a_o['levDistanceOrtho'].append(df["levDistHPCS_a_o"].corr(df["cossimHPCS_a_o"]))
        HPCS_a_o['levDistanceNormalizedOrtho'].append(df["levDistNormHPCS_a_o"].corr(df["cossimHPCS_a_o"]))
        HPCS_a_o['levDistancePhon'].append(df["levDistPhonHPCS_a_o"].corr(df["cossimHPCS_a_o"]))
        HPCS_a_o['levDistanceNormalizedPhon'].append(df["levDistNormPhonHPCS_a_o"].corr(df["cossimHPCS_a_o"]))

        HPCS_b_tr['simRating'].append(df["Response"].corr(df["cossimHPCS_b_tr"]))
        HPCS_b_tr['phonDistance'].append(df["distance_x"].corr(df["cossimHPCS_b_tr"]))
        HPCS_b_tr['levDistanceOrtho'].append(df["levDistHPCS_b_tr"].corr(df["cossimHPCS_b_tr"]))
        HPCS_b_tr['levDistanceNormalizedOrtho'].append(df["levDistNormHPCS_b_tr"].corr(df["cossimHPCS_b_tr"]))
        HPCS_b_tr['levDistancePhon'].append(df["levDistPhonHPCS_b_tr"].corr(df["cossimHPCS_b_tr"]))
        HPCS_b_tr['levDistanceNormalizedPhon'].append(df["levDistNormPhonHPCS_b_tr"].corr(df["cossimHPCS_b_tr"]))

        HPCS_b_o['simRating'].append(df["Response"].corr(df["cossimHPCS_b_o"]))
        HPCS_b_o['phonDistance'].append(df["distance_x"].corr(df["cossimHPCS_b_o"]))
        HPCS_b_o['levDistanceOrtho'].append(df["levDistHPCS_b_o"].corr(df["cossimHPCS_b_o"]))
        HPCS_b_o['levDistanceNormalizedOrtho'].append(df["levDistNormHPCS_b_o"].corr(df["cossimHPCS_b_o"]))
        HPCS_b_o['levDistancePhon'].append(df["levDistPhonHPCS_b_o"].corr(df["cossimHPCS_b_o"]))
        HPCS_b_o['levDistanceNormalizedPhon'].append(df["levDistNormPhonHPCS_b_o"].corr(df["cossimHPCS_b_o"]))

        HPI_c_tr['simRating'].append(df["Response"].corr(df["cossimHPI_c_tr"]))
        HPI_c_tr['phonDistance'].append(df["distance_x"].corr(df["cossimHPI_c_tr"]))
        HPI_c_tr['levDistanceOrtho'].append(df["levDistHPI_c_tr"].corr(df["cossimHPI_c_tr"]))
        HPI_c_tr['levDistanceNormalizedOrtho'].append(df["levDistNormHPI_c_tr"].corr(df["cossimHPI_c_tr"]))
        HPI_c_tr['levDistancePhon'].append(df["levDistPhonHPI_c_tr"].corr(df["cossimHPI_c_tr"]))
        HPI_c_tr['levDistanceNormalizedPhon'].append(df["levDistNormPhonHPI_c_tr"].corr(df["cossimHPI_c_tr"]))

        HPI_c_o['simRating'].append(df["Response"].corr(df["cossimHPI_c_o"]))
        HPI_c_o['phonDistance'].append(df["distance_x"].corr(df["cossimHPI_c_o"]))
        HPI_c_o['levDistanceOrtho'].append(df["levDistHPI_c_o"].corr(df["cossimHPI_c_o"]))
        HPI_c_o['levDistanceNormalizedOrtho'].append(df["levDistNormHPI_c_o"].corr(df["cossimHPI_c_o"]))
        HPI_c_o['levDistancePhon'].append(df["levDistPhonHPI_c_o"].corr(df["cossimHPI_c_o"]))
        HPI_c_o['levDistanceNormalizedPhon'].append(df["levDistNormPhonHPI_c_o"].corr(df["cossimHPI_c_o"]))

        HPI_a_tr['simRating'].append(df["Response"].corr(df["cossimHPI_a_tr"]))
        HPI_a_tr['phonDistance'].append(df["distance_x"].corr(df["cossimHPI_a_tr"]))
        HPI_a_tr['levDistanceOrtho'].append(df["levDistHPI_a_tr"].corr(df["cossimHPI_a_tr"]))
        HPI_a_tr['levDistanceNormalizedOrtho'].append(df["levDistNormHPI_a_tr"].corr(df["cossimHPI_a_tr"]))
        HPI_a_tr['levDistancePhon'].append(df["levDistPhonHPI_a_tr"].corr(df["cossimHPI_a_tr"]))
        HPI_a_tr['levDistanceNormalizedPhon'].append(df["levDistNormPhonHPI_a_tr"].corr(df["cossimHPI_a_tr"]))

        HPI_a_o['simRating'].append(df["Response"].corr(df["cossimHPI_a_o"]))
        HPI_a_o['phonDistance'].append(df["distance_x"].corr(df["cossimHPI_a_o"]))
        HPI_a_o['levDistanceOrtho'].append(df["levDistHPI_a_o"].corr(df["cossimHPI_a_o"]))
        HPI_a_o['levDistanceNormalizedOrtho'].append(df["levDistNormHPI_a_o"].corr(df["cossimHPI_a_o"]))
        HPI_a_o['levDistancePhon'].append(df["levDistPhonHPI_a_o"].corr(df["cossimHPI_a_o"]))
        HPI_a_o['levDistanceNormalizedPhon'].append(df["levDistNormPhonHPI_a_o"].corr(df["cossimHPI_a_o"]))

        HPI_b_tr['simRating'].append(df["Response"].corr(df["cossimHPI_b_tr"]))
        HPI_b_tr['phonDistance'].append(df["distance_x"].corr(df["cossimHPI_b_tr"]))
        HPI_b_tr['levDistanceOrtho'].append(df["levDistHPI_b_tr"].corr(df["cossimHPI_b_tr"]))
        HPI_b_tr['levDistanceNormalizedOrtho'].append(df["levDistNormHPI_b_tr"].corr(df["cossimHPI_b_tr"]))
        HPI_b_tr['levDistancePhon'].append(df["levDistPhonHPI_b_tr"].corr(df["cossimHPI_b_tr"]))
        HPI_b_tr['levDistanceNormalizedPhon'].append(df["levDistNormPhonHPI_b_tr"].corr(df["cossimHPI_b_tr"]))

        HPI_b_o['simRating'].append(df["Response"].corr(df["cossimHPI_b_o"]))
        HPI_b_o['phonDistance'].append(df["distance_x"].corr(df["cossimHPI_b_o"]))
        HPI_b_o['levDistanceOrtho'].append(df["levDistHPI_b_o"].corr(df["cossimHPI_b_o"]))
        HPI_b_o['levDistanceNormalizedOrtho'].append(df["levDistNormHPI_b_o"].corr(df["cossimHPI_b_o"]))
        HPI_b_o['levDistancePhon'].append(df["levDistPhonHPI_b_o"].corr(df["cossimHPI_b_o"]))
        HPI_b_o['levDistanceNormalizedPhon'].append(df["levDistNormPhonHPI_b_o"].corr(df["cossimHPI_b_o"]))

        

        


        


    #take the averages of the correlations
    for key in H_c_tr:
        H_c_tr[key] = np.mean(H_c_tr[key])

    for key in H_c_o:
        H_c_o[key] = np.mean(H_c_o[key])

    for key in H_a_tr:
        H_a_tr[key] = np.mean(H_a_tr[key])

    for key in H_a_o:
        H_a_o[key] = np.mean(H_a_o[key])

    for key in H_b_tr:
        H_b_tr[key] = np.mean(H_b_tr[key])

    for key in H_b_o:
        H_b_o[key] = np.mean(H_b_o[key])
    
    for key in PC_c_tr:
        PC_c_tr[key] = np.mean(PC_c_tr[key])

    for key in PC_c_o:
        PC_c_o[key] = np.mean(PC_c_o[key])

    for key in PC_a_tr:
        PC_a_tr[key] = np.mean(PC_a_tr[key])

    for key in PC_a_o:
        PC_a_o[key] = np.mean(PC_a_o[key])

    for key in PC_b_tr:
        PC_b_tr[key] = np.mean(PC_b_tr[key])

    for key in PC_b_o:
        PC_b_o[key] = np.mean(PC_b_o[key])

    for key in PCS_c_tr:
        PCS_c_tr[key] = np.mean(PCS_c_tr[key])

    for key in PCS_c_o:
        PCS_c_o[key] = np.mean(PCS_c_o[key])

    for key in PCS_a_tr:
        PCS_a_tr[key] = np.mean(PCS_a_tr[key])

    for key in PCS_a_o:
        PCS_a_o[key] = np.mean(PCS_a_o[key])

    for key in PCS_b_tr:
        PCS_b_tr[key] = np.mean(PCS_b_tr[key])

    for key in PCS_b_o:
        PCS_b_o[key] = np.mean(PCS_b_o[key])

    for key in PI_c_tr:
        PI_c_tr[key] = np.mean(PI_c_tr[key])

    for key in PI_c_o:
        PI_c_o[key] = np.mean(PI_c_o[key])

    for key in PI_a_tr:
        PI_a_tr[key] = np.mean(PI_a_tr[key])

    for key in PI_a_o:
        PI_a_o[key] = np.mean(PI_a_o[key])

    for key in PI_b_tr:
        PI_b_tr[key] = np.mean(PI_b_tr[key])

    for key in PI_b_o:
        PI_b_o[key] = np.mean(PI_b_o[key])

    for key in HPC_c_tr:
        HPC_c_tr[key] = np.mean(HPC_c_tr[key])

    for key in HPC_c_o:
        HPC_c_o[key] = np.mean(HPC_c_o[key])

    for key in HPC_a_tr:
        HPC_a_tr[key] = np.mean(HPC_a_tr[key])

    for key in HPC_a_o:
        HPC_a_o[key] = np.mean(HPC_a_o[key])

    for key in HPC_b_tr:
        HPC_b_tr[key] = np.mean(HPC_b_tr[key])

    for key in HPC_b_o:
        HPC_b_o[key] = np.mean(HPC_b_o[key])

    for key in HPCS_c_tr:
        HPCS_c_tr[key] = np.mean(HPCS_c_tr[key])

    for key in HPCS_c_o:
        HPCS_c_o[key] = np.mean(HPCS_c_o[key])

    for key in HPCS_a_tr:
        HPCS_a_tr[key] = np.mean(HPCS_a_tr[key])

    for key in HPCS_a_o:
        HPCS_a_o[key] = np.mean(HPCS_a_o[key])

    for key in HPCS_b_tr:
        HPCS_b_tr[key] = np.mean(HPCS_b_tr[key])

    for key in HPCS_b_o:
        HPCS_b_o[key] = np.mean(HPCS_b_o[key])

    for key in HPI_c_tr:
        HPI_c_tr[key] = np.mean(HPI_c_tr[key])

    for key in HPI_c_o:
        HPI_c_o[key] = np.mean(HPI_c_o[key])

    for key in HPI_a_tr:
        HPI_a_tr[key] = np.mean(HPI_a_tr[key])

    for key in HPI_a_o:
        HPI_a_o[key] = np.mean(HPI_a_o[key])

    for key in HPI_b_tr:
        HPI_b_tr[key] = np.mean(HPI_b_tr[key])

    for key in HPI_b_o:
        HPI_b_o[key] = np.mean(HPI_b_o[key])
    



    #print correlations for each parameter and measure
    w.writerow(["H_c_tr", H_c_tr['simRating'], H_c_tr['phonDistance'], H_c_tr['levDistanceOrtho'], H_c_tr['levDistanceNormalizedOrtho'], H_c_tr['levDistancePhon'], H_c_tr['levDistanceNormalizedPhon']])
    w.writerow(["H_c_o", H_c_o['simRating'], H_c_o['phonDistance'], H_c_o['levDistanceOrtho'], H_c_o['levDistanceNormalizedOrtho'], H_c_o['levDistancePhon'], H_c_o['levDistanceNormalizedPhon']])
    w.writerow(["H_a_tr", H_a_tr['simRating'], H_a_tr['phonDistance'], H_a_tr['levDistanceOrtho'], H_a_tr['levDistanceNormalizedOrtho'], H_a_tr['levDistancePhon'], H_a_tr['levDistanceNormalizedPhon']])
    w.writerow(["H_a_o", H_a_o['simRating'], H_a_o['phonDistance'], H_a_o['levDistanceOrtho'], H_a_o['levDistanceNormalizedOrtho'], H_a_o['levDistancePhon'], H_a_o['levDistanceNormalizedPhon']])
    w.writerow(["H_b_tr", H_b_tr['simRating'], H_b_tr['phonDistance'], H_b_tr['levDistanceOrtho'], H_b_tr['levDistanceNormalizedOrtho'], H_b_tr['levDistancePhon'], H_b_tr['levDistanceNormalizedPhon']])
    w.writerow(["H_b_o", H_b_o['simRating'], H_b_o['phonDistance'], H_b_o['levDistanceOrtho'], H_b_o['levDistanceNormalizedOrtho'], H_b_o['levDistancePhon'], H_b_o['levDistanceNormalizedPhon']])
    w.writerow(["PC_c_tr", PC_c_tr['simRating'], PC_c_tr['phonDistance'], PC_c_tr['levDistanceOrtho'], PC_c_tr['levDistanceNormalizedOrtho'], PC_c_tr['levDistancePhon'], PC_c_tr['levDistanceNormalizedPhon']])
    w.writerow(["PC_c_o", PC_c_o['simRating'], PC_c_o['phonDistance'], PC_c_o['levDistanceOrtho'], PC_c_o['levDistanceNormalizedOrtho'], PC_c_o['levDistancePhon'], PC_c_o['levDistanceNormalizedPhon']])
    w.writerow(["PC_a_tr", PC_a_tr['simRating'], PC_a_tr['phonDistance'], PC_a_tr['levDistanceOrtho'], PC_a_tr['levDistanceNormalizedOrtho'], PC_a_tr['levDistancePhon'], PC_a_tr['levDistanceNormalizedPhon']])
    w.writerow(["PC_a_o", PC_a_o['simRating'], PC_a_o['phonDistance'], PC_a_o['levDistanceOrtho'], PC_a_o['levDistanceNormalizedOrtho'], PC_a_o['levDistancePhon'], PC_a_o['levDistanceNormalizedPhon']])
    w.writerow(["PC_b_tr", PC_b_tr['simRating'], PC_b_tr['phonDistance'], PC_b_tr['levDistanceOrtho'], PC_b_tr['levDistanceNormalizedOrtho'], PC_b_tr['levDistancePhon'], PC_b_tr['levDistanceNormalizedPhon']])
    w.writerow(["PC_b_o", PC_b_o['simRating'], PC_b_o['phonDistance'], PC_b_o['levDistanceOrtho'], PC_b_o['levDistanceNormalizedOrtho'], PC_b_o['levDistancePhon'], PC_b_o['levDistanceNormalizedPhon']])
    w.writerow(["PCS_c_tr", PCS_c_tr['simRating'], PCS_c_tr['phonDistance'], PCS_c_tr['levDistanceOrtho'], PCS_c_tr['levDistanceNormalizedOrtho'], PCS_c_tr['levDistancePhon'], PCS_c_tr['levDistanceNormalizedPhon']])
    w.writerow(["PCS_c_o", PCS_c_o['simRating'], PCS_c_o['phonDistance'], PCS_c_o['levDistanceOrtho'], PCS_c_o['levDistanceNormalizedOrtho'], PCS_c_o['levDistancePhon'], PCS_c_o['levDistanceNormalizedPhon']])
    w.writerow(["PCS_a_tr", PCS_a_tr['simRating'], PCS_a_tr['phonDistance'], PCS_a_tr['levDistanceOrtho'], PCS_a_tr['levDistanceNormalizedOrtho'], PCS_a_tr['levDistancePhon'], PCS_a_tr['levDistanceNormalizedPhon']])
    w.writerow(["PCS_a_o", PCS_a_o['simRating'], PCS_a_o['phonDistance'], PCS_a_o['levDistanceOrtho'], PCS_a_o['levDistanceNormalizedOrtho'], PCS_a_o['levDistancePhon'], PCS_a_o['levDistanceNormalizedPhon']])
    w.writerow(["PCS_b_tr", PCS_b_tr['simRating'], PCS_b_tr['phonDistance'], PCS_b_tr['levDistanceOrtho'], PCS_b_tr['levDistanceNormalizedOrtho'], PCS_b_tr['levDistancePhon'], PCS_b_tr['levDistanceNormalizedPhon']])
    w.writerow(["PCS_b_o", PCS_b_o['simRating'], PCS_b_o['phonDistance'], PCS_b_o['levDistanceOrtho'], PCS_b_o['levDistanceNormalizedOrtho'], PCS_b_o['levDistancePhon'], PCS_b_o['levDistanceNormalizedPhon']])
    w.writerow(["PI_c_tr", PI_c_tr['simRating'], PI_c_tr['phonDistance'], PI_c_tr['levDistanceOrtho'], PI_c_tr['levDistanceNormalizedOrtho'], PI_c_tr['levDistancePhon'], PI_c_tr['levDistanceNormalizedPhon']])
    w.writerow(["PI_c_o", PI_c_o['simRating'], PI_c_o['phonDistance'], PI_c_o['levDistanceOrtho'], PI_c_o['levDistanceNormalizedOrtho'], PI_c_o['levDistancePhon'], PI_c_o['levDistanceNormalizedPhon']])
    w.writerow(["PI_a_tr", PI_a_tr['simRating'], PI_a_tr['phonDistance'], PI_a_tr['levDistanceOrtho'], PI_a_tr['levDistanceNormalizedOrtho'], PI_a_tr['levDistancePhon'], PI_a_tr['levDistanceNormalizedPhon']])
    w.writerow(["PI_a_o", PI_a_o['simRating'], PI_a_o['phonDistance'], PI_a_o['levDistanceOrtho'], PI_a_o['levDistanceNormalizedOrtho'], PI_a_o['levDistancePhon'], PI_a_o['levDistanceNormalizedPhon']])
    w.writerow(["PI_b_tr", PI_b_tr['simRating'], PI_b_tr['phonDistance'], PI_b_tr['levDistanceOrtho'], PI_b_tr['levDistanceNormalizedOrtho'], PI_b_tr['levDistancePhon'], PI_b_tr['levDistanceNormalizedPhon']])
    w.writerow(["PI_b_o", PI_b_o['simRating'], PI_b_o['phonDistance'], PI_b_o['levDistanceOrtho'], PI_b_o['levDistanceNormalizedOrtho'], PI_b_o['levDistancePhon'], PI_b_o['levDistanceNormalizedPhon']])
    w.writerow(["HPC_c_tr", HPC_c_tr['simRating'], HPC_c_tr['phonDistance'], HPC_c_tr['levDistanceOrtho'], HPC_c_tr['levDistanceNormalizedOrtho'], HPC_c_tr['levDistancePhon'], HPC_c_tr['levDistanceNormalizedPhon']])
    w.writerow(["HPC_c_o", HPC_c_o['simRating'], HPC_c_o['phonDistance'], HPC_c_o['levDistanceOrtho'], HPC_c_o['levDistanceNormalizedOrtho'], HPC_c_o['levDistancePhon'], HPC_c_o['levDistanceNormalizedPhon']])
    w.writerow(["HPC_a_tr", HPC_a_tr['simRating'], HPC_a_tr['phonDistance'], HPC_a_tr['levDistanceOrtho'], HPC_a_tr['levDistanceNormalizedOrtho'], HPC_a_tr['levDistancePhon'], HPC_a_tr['levDistanceNormalizedPhon']])
    w.writerow(["HPC_a_o", HPC_a_o['simRating'], HPC_a_o['phonDistance'], HPC_a_o['levDistanceOrtho'], HPC_a_o['levDistanceNormalizedOrtho'], HPC_a_o['levDistancePhon'], HPC_a_o['levDistanceNormalizedPhon']])
    w.writerow(["HPC_b_tr", HPC_b_tr['simRating'], HPC_b_tr['phonDistance'], HPC_b_tr['levDistanceOrtho'], HPC_b_tr['levDistanceNormalizedOrtho'], HPC_b_tr['levDistancePhon'], HPC_b_tr['levDistanceNormalizedPhon']])
    w.writerow(["HPC_b_o", HPC_b_o['simRating'], HPC_b_o['phonDistance'], HPC_b_o['levDistanceOrtho'], HPC_b_o['levDistanceNormalizedOrtho'], HPC_b_o['levDistancePhon'], HPC_b_o['levDistanceNormalizedPhon']])
    w.writerow(["HPCS_c_tr", HPCS_c_tr['simRating'], HPCS_c_tr['phonDistance'], HPCS_c_tr['levDistanceOrtho'], HPCS_c_tr['levDistanceNormalizedOrtho'], HPCS_c_tr['levDistancePhon'], HPCS_c_tr['levDistanceNormalizedPhon']])
    w.writerow(["HPCS_c_o", HPCS_c_o['simRating'], HPCS_c_o['phonDistance'], HPCS_c_o['levDistanceOrtho'], HPCS_c_o['levDistanceNormalizedOrtho'], HPCS_c_o['levDistancePhon'], HPCS_c_o['levDistanceNormalizedPhon']])
    w.writerow(["HPCS_a_tr", HPCS_a_tr['simRating'], HPCS_a_tr['phonDistance'], HPCS_a_tr['levDistanceOrtho'], HPCS_a_tr['levDistanceNormalizedOrtho'], HPCS_a_tr['levDistancePhon'], HPCS_a_tr['levDistanceNormalizedPhon']])
    w.writerow(["HPCS_a_o", HPCS_a_o['simRating'], HPCS_a_o['phonDistance'], HPCS_a_o['levDistanceOrtho'], HPCS_a_o['levDistanceNormalizedOrtho'], HPCS_a_o['levDistancePhon'], HPCS_a_o['levDistanceNormalizedPhon']])
    w.writerow(["HPCS_b_tr", HPCS_b_tr['simRating'], HPCS_b_tr['phonDistance'], HPCS_b_tr['levDistanceOrtho'], HPCS_b_tr['levDistanceNormalizedOrtho'], HPCS_b_tr['levDistancePhon'], HPCS_b_tr['levDistanceNormalizedPhon']])
    w.writerow(["HPCS_b_o", HPCS_b_o['simRating'], HPCS_b_o['phonDistance'], HPCS_b_o['levDistanceOrtho'], HPCS_b_o['levDistanceNormalizedOrtho'], HPCS_b_o['levDistancePhon'], HPCS_b_o['levDistanceNormalizedPhon']])
    w.writerow(["HPI_c_tr", HPI_c_tr['simRating'], HPI_c_tr['phonDistance'], HPI_c_tr['levDistanceOrtho'], HPI_c_tr['levDistanceNormalizedOrtho'], HPI_c_tr['levDistancePhon'], HPI_c_tr['levDistanceNormalizedPhon']])
    w.writerow(["HPI_c_o", HPI_c_o['simRating'], HPI_c_o['phonDistance'], HPI_c_o['levDistanceOrtho'], HPI_c_o['levDistanceNormalizedOrtho'], HPI_c_o['levDistancePhon'], HPI_c_o['levDistanceNormalizedPhon']])
    w.writerow(["HPI_a_tr", HPI_a_tr['simRating'], HPI_a_tr['phonDistance'], HPI_a_tr['levDistanceOrtho'], HPI_a_tr['levDistanceNormalizedOrtho'], HPI_a_tr['levDistancePhon'], HPI_a_tr['levDistanceNormalizedPhon']])
    w.writerow(["HPI_a_o", HPI_a_o['simRating'], HPI_a_o['phonDistance'], HPI_a_o['levDistanceOrtho'], HPI_a_o['levDistanceNormalizedOrtho'], HPI_a_o['levDistancePhon'], HPI_a_o['levDistanceNormalizedPhon']])
    w.writerow(["HPI_b_tr", HPI_b_tr['simRating'], HPI_b_tr['phonDistance'], HPI_b_tr['levDistanceOrtho'], HPI_b_tr['levDistanceNormalizedOrtho'], HPI_b_tr['levDistancePhon'], HPI_b_tr['levDistanceNormalizedPhon']])
    w.writerow(["HPI_b_o", HPI_b_o['simRating'], HPI_b_o['phonDistance'], HPI_b_o['levDistanceOrtho'], HPI_b_o['levDistanceNormalizedOrtho'], HPI_b_o['levDistancePhon'], HPI_b_o['levDistanceNormalizedPhon']])

