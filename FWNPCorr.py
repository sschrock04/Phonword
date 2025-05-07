import phonword_python3 as phonword
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

elp = pd.read_csv("C:\\Users\\Stephen\\Downloads\\elp_data_for_jolreactivity.csv")
initConsMat = pd.read_csv("C:\\Users\\Stephen\\Downloads\\initialConsConfMat+5.csv")
vowelsMat = pd.read_csv("C:\\Users\\Stephen\\Downloads\\vowelsConfMat+5.csv")
finalConsMat = pd.read_csv("C:\\Users\\Stephen\\Downloads\\finalConsConfMat+5.csv")
print(initConsMat)
print(vowelsMat)
print(finalConsMat)

#def phonProb
