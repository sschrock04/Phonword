""""
the following code can be replicated to get the cosine similarity between two word representations (PhonWordRep can be replaced by HoloWord, IPAWordRep, PhonWordRepStress along with their respective makerep functions)
myWordGen = PhonWordRep(d = 1024, ngramType = "tr", vis_scale = [1, 2], spaces = True, minsize = 0, bindOp = "convolution")
vectorRep1 = myWordGen.make_repPhon('their')
vectorRep2 = myWordGen.make_repPhon('there')
print(cosSim(vectorRep1, vectorRep2))
"""
from ctypes import ArgumentError
import numpy
from functools import reduce
from numpy.linalg import norm
import numpy as np
from g2p_en import G2p 
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '_', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
casedAlphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
phonemes = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', '_']
phonemesStress = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B0', 'B1', 'B2', 'CH0', 'CH1', 'CH2', 'D0', 'D1', 'D2', 'DH0', 'DH1', 'DH2', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F0', 'F1', 'F2', 'G0', 'G1', 'G2', 'HH0', 'HH1', 'HH2', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH0', 'JH1', 'JH2', 'K0', 'K1', 'K2', 'L0', 'L1', 'L2', 'M0', 'M1', 'M2', 'N0', 'N1', 'N2', 'NG0', 'NG1', 'NG2', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P0', 'P1', 'P2', 'R0', 'R1', 'R2', 'S0', 'S1', 'S2', 'SH0', 'SH1', 'SH2', 'T0', 'T1', 'T2', 'TH0', 'TH1', 'TH2', 'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2', 'V0', 'V1', 'V2', 'W0', 'W1', 'W2', 'Y0', 'Y1', 'Y2', 'Z0', 'Z1', 'Z2', 'ZH0', 'ZH1', 'ZH2', '_']
stressVowels = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2']

IPAPhons = ['lbv', 'fnt', 'rzd', 'pla', 'stp', 'glt', 'rnd', 'pal', 'bck', 'vel', 'vcd', 'lmd', 'nas', 'vwl', 'hgh', 'mid', 'cnt', 'end', 'smh', 'umd', 'low', 'lat', 'unr', 'apr', 'lbd', 'frc', 'blb', 'dnt', 'vls', 'alv', 'beg', '_']

phone_feature_map = {
    'M': ['blb', 'nas'],
    'P': ['vls', 'blb', 'stp'],
    'B': ['vcd', 'blb', 'stp'],
    'F': ['vls', 'lbd', 'frc'],
    'V': ['vcd', 'lbd', 'frc'],
    'TH': ['vls', 'dnt', 'frc'],
    'DH': ['vcd', 'dnt', 'frc'],
    'N': ['alv', 'nas'],
    'T': ['vls', 'alv', 'stp'],
    'D': ['vcd', 'alv', 'stp'],
    'S': ['vls', 'alv', 'frc'],
    'Z': ['vcd', 'alv', 'frc'],
    'R': ['alv', 'apr'],
    'L': ['alv', 'lat'],
    'SH': ['vls', 'pla', 'frc'],
    'ZH': ['vcd', 'pla', 'frc'],
    'Y': ['pal', 'apr'],
    'NG': ['vel', 'nas'],
    'K': ['vls', 'vel', 'stp'],
    'G': ['vcd', 'vel', 'stp'],
    'W': ['lbv', 'apr'],
    'HH': ['glt', 'apr'],
    'CH': ['vls', 'alv', 'stp', 'frc'],
    'JH': ['vcd', 'alv', 'stp', 'frc'],
    'AO': ['lmd', 'bck', 'rnd', 'vwl'],
    'AA': ['low', 'bck', 'unr', 'vwl'],
    'IY': ['hgh', 'fnt', 'unr', 'vwl'],
    'UW': ['hgh', 'bck', 'rnd', 'vwl'],
    'EH': ['lmd', 'fnt', 'unr', 'vwl'],
    'IH': ['smh', 'fnt', 'unr', 'vwl'],
    'UH': ['smh', 'bck', 'rnd', 'vwl'],
    'AH': ['mid', 'cnt', 'unr', 'vwl'],
    'AE': ['low', 'fnt', 'unr', 'vwl'],
    'EY': ['lmd', 'smh', 'fnt', 'unr', 'vwl'],
    'AY': ['low', 'smh', 'fnt', 'cnt', 'unr', 'vwl'],
    'OW': ['umd', 'smh', 'bck', 'rnd', 'vwl'],
    'AW': ['low', 'smh', 'bck', 'cnt', 'unr', 'rnd', 'vwl'],
    'OY': ['lmd', 'smh', 'bck', 'fnt', 'rnd', 'unr', 'vwl'],
    'ER': ['umd', 'cnt', 'rzd', 'vwl'],
    '_':['_'],
    '^': ['beg',],
    '$': ['end',]
}


for i in range(len(phonemes)):
    phonemes[i] = phonemes[i].lower()

for i in range(len(phonemesStress)):
    phonemesStress[i] = phonemesStress[i].lower()


def hammingSim(a, b):
    '''
    Computes the normalized Hamming similarity between binary vectors a and b.
    '''
    h = float(numpy.sum((a > 0) * (b > 0))) / float(numpy.sum((a > 0) + (b > 0)))
    return h

def normalize(a):
    '''
    Normalize a vector to length 1.
    '''
    return a / numpy.sum(a**2.0)**0.5

def maj(args, p = .5):
    '''
    The majority-rule operation for binary vectors.
    '''
    if len(args) == 0:
        raise ArgumentError('Need something to work with!')
    if len(args) == 1:
        argSum = args[0]
    else:
        argSum = reduce(lambda a,b: a+b, args)
        argSum[argSum == -2*p + 1] = (numpy.roll(args[0], 1) * numpy.roll(args[len(args)-1], 1))[argSum == -2*p + 1]
    
    argSum[argSum < -2*p + 1] = -1.0
    argSum[argSum > -2*p + 1] = 1.0
    
    return argSum
    
def xor(a, b):
    '''
    The X-OR operation for binary (-1 or 1) vectors.
    '''
    return -(a * b)

def entropy(p):
    '''
    Compute the entropy of a vector p of non-negative numbers (normalized to sum
    to 1 and thereby be probabilities).
    '''
    p /= numpy.sum(p)
    return -numpy.dot(p[p > 0], numpy.log(p[p > 0]))

def cconv(a, b):
    '''
    Computes the circular convolution of the (real-valued) vectors a and b.
    '''
    return numpy.fft.ifft(numpy.fft.fft(a) * numpy.fft.fft(b)).real

def ccorr(a, b):
    '''
    Computes the circular correlation (inverse convolution) of the real-valued
    vector a with b.
    '''
    return cconv(numpy.roll(a[::-1], 1), b)

def convpow(a, p):
    '''
    Computes the convolutive power of the real-valued vector a, to the
    (real-valued) power p.
    '''
    return numpy.fft.ifft(numpy.fft.fft(a)**p).real

def cosine(a,b):
    '''
    Computes the cosine of the angle between the vectors a and b.
    '''
    sumSqA = numpy.sum(a**2.0)
    sumSqB = numpy.sum(b**2.0)
    if sumSqA == 0.0 or sumSqB == 0.0: return 0.0
    return numpy.dot(a,b) * (sumSqA * sumSqB)**-0.5

def euclidean_distance(a,b):
    '''
    Return the Euclidean distance between vectors a and b.
    '''
    return numpy.sum((a - b)**2.0)**0.5

def dl(str1, str2):
    #divide by longer string length
    '''
    Computes the Damerau-Levenshtein distance between the two given strings.
    '''
    f = numpy.zeros((len(str1) + 1,len(str2) + 1), dtype='int')
    cost = 0
    
    for i in range(1, f.shape[0]): f[i][0] = i
    for j in range(1, f.shape[1]): f[0][j] = j
    
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]: cost = 0
            else: cost = 1
            f[i][j] = min(f[i - 1][j - 1] + cost, f[i - 1][j] + 1, f[i][j - 1] + 1)
            if i > 1 and j > 1 and str1[i - 1] == str2[j - 2] and str1[i - 2] == str2[j - 1]:
                f[i][j] = min(f[i][j], f[i - 2, j - 2] + cost)
    
    return f[f.shape[0]-1][f.shape[1]-1]


def dlNormalized(str1, str2):
    #divide by longer string length
    '''
    Computes the Damerau-Levenshtein distance between the two given strings.
    '''
    f = numpy.zeros((len(str1) + 1,len(str2) + 1), dtype='int')
    cost = 0
    if (len(str1) >= len(str2)):
        long = len(str1)
    else:
        long = len(str2)
    
    for i in range(1, f.shape[0]): f[i][0] = i
    for j in range(1, f.shape[1]): f[0][j] = j
    
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]: cost = 0
            else: cost = 1
            f[i][j] = min(f[i - 1][j - 1] + cost, f[i - 1][j] + 1, f[i][j - 1] + 1)
            if i > 1 and j > 1 and str1[i - 1] == str2[j - 2] and str1[i - 2] == str2[j - 1]:
                f[i][j] = min(f[i][j], f[i - 2, j - 2] + cost)
    
    return (f[f.shape[0]-1][f.shape[1]-1]) / long

def ld(str1, str2):
    '''
    Computes the Levenshtein distance between the two given strings.
    '''
    f = numpy.zeros((len(str1) + 1,len(str2) + 1), dtype='int')
    cost = 0
    
    for i in range(1, f.shape[0]): f[i][0] = i
    for j in range(1, f.shape[1]): f[0][j] = j
    
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]: cost = 0
            else: cost = 1
            f[i][j] = min(f[i - 1][j - 1] + cost, f[i - 1][j] + 1, f[i][j - 1] + 1)
    
    return f[f.shape[0]-1][f.shape[1]-1]

def ordConv(a, b, p1, p2):
    '''
    Performs ordered (non-commutative) circular convolution on the vectors a and
    b by first permuting them according to the index vectors p1 and p2.
    '''
    return cconv(a[p1], b[p2])

def convBind(p1, p2, l):
    '''
    Given a list of vectors, iteratively convolves them into a single vector
    (i.e., "binds" them together).
    '''
    return reduce(lambda a,b: normalize(ordConv(a, b, p1, p2)), l)

def slotBind(p, l):
    toReturn = numpy.zeros_like(p)
    for i, item in enumerate(l): toReturn += normalize(cconv(item, convpow(p, i+1.0)))# + normalize(cconv(item, convpow(p, float(i - len(l)))))
    return normalize(toReturn)

def spatialBind(p, l):
    toReturn = numpy.zeros_like(p)
    for i, item in enumerate(l): toReturn += normalize(cconv(item, convpow(p, float(i+1) / float(len(l)))))
    return normalize(toReturn)

def addBind(p1, p2, l):
    '''
    Given a list of vectors, binds them together by iteratively convolving them
    with place vectors and adding them up.
    '''
    return normalize(reduce(lambda a,b: cconv(a, p1) + cconv(b, p2), l))

def bscBind(p1, p2, l, p = .5):
    '''
    Given a list of binary (-1 or 1) vectors, binds them together by computing
    their XOR with place vectors and adding them up.  This is done in a pairwise
    fashion, i.e., ((a+b)+(c+d))+((e+f)+(g+h)), to balance the added noise across
    all components equally.
    '''
    tempElements = [elem for elem in l]
    
    while len(tempElements) > 1:
        newElements = []
        for i in range(0, len(tempElements)-1, 2):
            if i+2 == len(tempElements)-1:
                newElements.append(maj([xor(maj([xor(tempElements[i], p1), xor(tempElements[i+1], p2)], p), p1), xor(tempElements[i+2], p2)], p))
            else:
                newElements.append(maj([xor(tempElements[i], p1), xor(tempElements[i+1], p2)], p))
        tempElements = [elem for elem in newElements]
    
    return tempElements[0]

def getOpenNGrams(seg, scale, spaces, minsize):
    '''
    Returns a list of the open n-grams of the string "seg", with sizes specified
    by "scale", which should be a list of positive integers in ascending order.
    "Spaces" indicates whether a space character should be used to mark gaps in
    non-contiguous n-grams.
    '''
    ngrams = []
    
    for size in scale:
        if size > len(seg): break
        
        for i in range(len(seg)):
            if i+size > len(seg): break
            if size > minsize or len(seg) <= minsize: ngrams.append(seg[i:i+size])
            if i+size == len(seg): continue
            for b in range(1, size):
                for e in range(1, len(seg)-i-size+1):
                    ngrams.append(seg[i:i+b]+('_' if spaces else '')+seg[i+b+e:i+e+size])
    
    return ngrams

def getTRNGrams(seg, scale, spaces, minsize):
    '''
    Returns a list of n-grams from the string "seg", according to the "terminal
    relative" (TR) encoding procedure, i.e., any internal n-gram gets included
    both by itself and as part of a non-contiguous n-gram with n-grams at either
    end of "seg".  "Scale" is a list of ascending integers reflecting the sizes
    of the n-gram chunks.  "Spaces" indicates whether a space character should
    be used to mark gaps in non-contiguous n-grams.
    '''
    ngrams = []
    
    for size in scale:
        if size > len(seg): break
        
        for i in range(len(seg)-size+1):
            if '_' in seg[i:i+size]: continue
            
            if size > minsize or len(seg) <= minsize: ngrams.append(seg[i:i+size])
            if seg[0] != '_':
                for fsize in scale:
                    if '_' in seg[:fsize] or fsize > i or (i==fsize and fsize+size in scale): break
                    ngrams.append(seg[:fsize] + ('_' if spaces and i > fsize else '') + seg[i:i+size])
            if seg[-1] != '_':
                for fsize in scale:
                    if '_' in seg[-fsize:] or i + size > len(seg)-fsize or (i+size==len(seg)-fsize and fsize+size in scale): break
                    ngrams.append(seg[i:i+size] + ('_' if spaces and i+size < len(seg)-fsize else '') + seg[-fsize:])
    
    return ngrams

class HoloWordRep:
    '''
    Spawns objects capable of generating holographic word-form representations
    in both the visual and auditory modalities. Auditory representations depend
    on the given word being present in the CMU Pronouncing Dictionary.
    '''
    
    def __init__(self, d = 1024, ngramType = 'tr', vis_scale=[1,2], spaces = True, minsize = 0, bindOp = 'convolution', seed = None):
        '''
        Creates a new holographic word-form representation generator.
        d - dimensionality of the representations (typically quite large)
        ngramType - how words are to be decomposed into n-grams; either 'tr'
            (terminal relative) or 'open'.
        vis_scale - a list of the size of successive chunks in visual word
            representations (in # of letters)
        spaces - whether or not to use a space character to mark gaps in non-
            contiguous n-grams
        bindOp - one of "convolution", "addition", or "bsc" (or just the initial
            letter of those) indicating the method by which n-grams should be
            bound together
        seed - integer seed for the numpy random number generator (optional)
        '''
        
        numpy.random.seed(seed)
        
        self.d = d
        self.vis_scale = sorted(vis_scale)
        self.spaces = spaces
        
        # Set the n-gram extraction method
        if ngramType.lower().startswith('t'):
            self.getNGrams = lambda s: getTRNGrams(s, self.vis_scale, self.spaces, minsize)
        elif ngramType.lower().startswith('o'):
            self.getNGrams = lambda s: getOpenNGrams(s, self.vis_scale, self.spaces, minsize)
        else:
            raise ArgumentError('Invalid n-gram type!')
        
        bindOp = bindOp.lower()
        
        if bindOp.startswith('c') or bindOp.startswith('a'):
            # Create random vectors representing individual letters, the "atoms"
            # of the word representations.
            self.letters = dict(list(zip(alphabet, [normalize(numpy.random.randn(d) * d**-0.5) for letter in alphabet])))
            
            self.chunk = lambda l: normalize(reduce(lambda a,b: a+b, l, numpy.zeros((d))))    #Chunking is done by superposition (vector addition)
            
            if bindOp.startswith('c'):
                # Permutation operators that scramble the vectors in a
                # convolution operation; this makes the operation non-commutative and
                # thus allows it to encode order.
                self.place1 = numpy.random.permutation(d)
                self.place2 = numpy.random.permutation(d)
                
                self.invplace1 = numpy.zeros((d), dtype='int')
                self.invplace2 = numpy.zeros((d), dtype='int')
                
                for i in range(d):
                    self.invplace1[self.place1[i]] = i
                    self.invplace2[self.place2[i]] = i
                
                self.bind = lambda l: convBind(self.place1, self.place2, l)
            else:
                # Here, to make binding non-commutative, each operand is first
                # convolved with a place vector, and the two results are added.
                # (Results in "fuzzy position" coding)
                self.place1 = normalize(numpy.random.randn(d) * d**-0.5)
                self.place2 = normalize(numpy.random.randn(d) * d**-0.5)
                

                
                self.bind = lambda l: addBind(self.place1, self.place2, l)
        elif bindOp.startswith('b'):
            # Create random vectors representing individual letters, the "atoms"
            # of the word representations.  These are binary vectors taking the
            # values -1 or 1.
            self.letters = dict(list(zip(alphabet, [(numpy.random.rand(d) < 0.5) * 2.0 - 1.0 for letter in alphabet])))
            
            self.place1 = (numpy.random.rand(d) < 0.5) * 2.0 - 1.0
            self.place2 = (numpy.random.rand(d) < 0.5) * 2.0 - 1.0
            
            self.chunk = lambda l: maj(l, 0.5)
            self.bind = lambda l: bscBind(self.place1, self.place2, l, 0.5)
        elif bindOp.startswith('sl'):
            self.getNGrams = lambda s: [s]
            self.letters = dict(list(zip(alphabet, [normalize(numpy.random.randn(d) * d**-0.5) for letter in alphabet])))
            self.place = normalize(numpy.random.randn(d) * d**-0.5)
            
            self.chunk = lambda l: normalize(reduce(lambda a,b: a+b, l, numpy.zeros((d))))    #Chunking is done by superposition (vector addition)
            self.bind = lambda l: slotBind(self.place, l)
        elif bindOp.startswith('sp'):
            self.getNGrams = lambda s: [s]
            self.letters = dict(list(zip(alphabet, [normalize(numpy.random.randn(d) * d**-0.5) for letter in alphabet])))
            self.place = normalize(numpy.random.randn(d) * d**-0.5)
            
            self.chunk = lambda l: normalize(reduce(lambda a,b: a+b, l, numpy.zeros((d))))    #Chunking is done by superposition (vector addition)
            self.bind = lambda l: spatialBind(self.place, l)
        else:
            raise ArgumentError('Invalid binding operator specification!')
    
    def make_rep(self, word):
        '''
        Returns a holographic representation of the given word. The word may be
        given as just a string, or as a list of lists, where each sublist is a
        way of segmenting the word, e.g., "homework" vs. [['homework'], ['home',
        'work']].
        '''
        
        if type(word) == type(''): word = [ [word] ]
        
        # Create a visual word form representation based on the letters present
        # in the word.
        formReps = []
        for form in word:
            segReps = []
            
            for seg in form:
                seg = seg.strip().lower()
                ngrams = self.getNGrams(seg)
                segRep = self.chunk([self.bind([self.letters[l] for l in ngram]) for ngram in ngrams])
                segReps.append(segRep)
            
            if len(segReps) > 1:
                segChunks = []
                for size in range(1,len(segReps)):
                    for i in range(len(segReps)-size+1):
                        segChunks.append(self.bind(segReps[i:(i+size)]))
            else:
                segChunks = [segReps[0]]
            formReps.append(self.chunk(segChunks))
        return self.chunk(formReps)

def EvalConstraints(n = 1, **repArgs):
    '''
    Evaluates the constraints from Hannagan, et al., (in press). These are
    based on the relative amount of facilitation from prime to target, which
    should correlate with greater similarity of the underlying word-form
    representations.  Similarities are averaged over "n" simulations.  Returns
    the estimated similarites, their standard deviations, and a vector of
    booleans reflecting whether or not that constraint was satisfied.
    "**repArgs" arguments get passed to the word-form generator.
    '''
    pairs = [('abcde', 'abcde'), ('abde', 'abcde'), ('abccde', 'abcde'), ('abcfde', 'abcde'), ('abfge', 'abcde'), ('afcde', 'abcde'), ('abgdef', 'abcdef'), ('abgdhf', 'abcdef'), ('fbcde', 'abcde'), ('abfde', 'abcde'), ('abcdf', 'abcde'), ('abdce', 'abcde'), ('badcfehg', 'abcdefgh'), ('abedcf', 'abcdef'), ('acfde', 'abcde'), ('abcde', 'abcdefg'), ('cdefg', 'abcdefg'), ('acdeg', 'abcdefg'), ('abcbef', 'abcbdef'), ('abcdef', 'abcbdef')]
    sim = numpy.zeros((len(pairs)))
    simSq = numpy.zeros((len(pairs)))
    
    for i in range(n):
        h = HoloWordRep(**repArgs)
        newData = numpy.array([cosine(h.make_rep(prime), h.make_rep(target)) for prime, target in pairs])
        sim += newData
        simSq += newData**2.0
    
    sim /= float(n)
    sd = ((simSq - n * sim**2.0) / float(n - 1))**0.5
    
    trends = [sim[0] == numpy.max(sim), sim[1] < sim[0], sim[2] < sim[0], sim[3] < sim[0], sim[4] < sim[0], sim[5] < sim[0], sim[6] < sim[0], sim[7] < sim[6], sim[8] < sim[9], sim[9] < sim[0], sim[10] < sim[9], sim[11] > sim[4], sim[12] == numpy.min(sim), sim[13] < sim[6] and sim[13] > sim[7], sim[14] < sim[5], sim[15] > numpy.min(sim), sim[16] > numpy.min(sim), sim[17] > numpy.min(sim), sim[18] > numpy.min(sim), numpy.abs(sim[19] - sim[18]) == numpy.min(numpy.abs(sim[19] - sim[:19]))]
    
    return sim, sd, trends

def Substitutions(length = 7, numsims = 1, **repArgs):
    '''
    For words of "length" unique letters, computes the similarity resulting
    from replacing a letter at each location.  Similarities (and their standard
    deviations) are computed over "numsims" simulations.  "**repArgs" get passed
    to the representation generator.
    '''
    target = ''.join(map(chr, list(range(97, 97+min(length, 25)))))
    primes = [target[:i]+'z'+target[i+1:] for i in range(len(target))]
    sim = numpy.zeros((len(primes)))
    simSq = numpy.zeros((len(primes)))
    
    for n in range(numsims):
        h = HoloWordRep(**repArgs)
        
        targetRep = h.make_rep(target)
        newData = numpy.array([cosine(h.make_rep(prime), targetRep) for prime in primes])
        sim += newData
        simSq += newData**2.0
    
    sim /= float(numsims)
    sd = ((simSq - numsims * sim**2.0) / float(numsims - 1))**0.5
    
    return sim, sd

def MakeReps(words, segment = False, filename = None, **repArgs):
    '''
    Makes representations for the words in "words", which can be a list of strings
    or a string giving a filename.
    segment - Whether or not to find ways of segmenting each word into other,
        shorter words (e.g., "homework" = "home" + "work")
    filename - A filename to which the representations can be written in CSV format
    repArgs - named arguments to be passed to HoloWordRep
    '''
    if type(words) == type(''):
        word_list = []
        FIN = open(words, 'r')
        for line in FIN:
            line = line.strip().lower()
            line = line.split()
            word_list.append(line[0])
        FIN.close()
        words = word_list
        
    h = HoloWordRep(**repArgs)
    
    if segment:
        forms = SegmentWords(words)
        reps = numpy.array([h.make_rep(form) for form in forms])
    else:
        reps = numpy.array([h.make_rep(word) for word in words])            # Make representations for all words in the list
    
    if filename != None:
        FOUT = open(filename, 'w')
        for i, word in enumerate(words):
            FOUT.write(word + ',' + ','.join([str(x) for x in reps[i]]) + '\n')
        FOUT.close()
    
    return reps

def SimMatrix(words, numsims=1, numToCompare = None, segment = False, **repArgs):
    '''
    Returns a matrix containing cosine similarities between word-form representations
    of the words given in "words" (either a list of strings or a filename).  Similarities
    are averaged over "numsims" simulations.
    numToCompare - If specified, the matrix only contains the top "numToCompare"
        similarity values
    segment - Whether or not to find ways of segmenting each word into other,
        shorter words (e.g., "homework" = "home" + "work")
    repArgs - named arguments to be passed to HoloWordRep
    '''
    if type(words) == type(''):
        FIN = open(words, 'r')
        words = [line.strip() for line in FIN]
        FIN.close()
    
    if numToCompare == None:
        sim = numpy.zeros((len(words), len(words)))
    else:
        sim = numpy.zeros((len(words), numToCompare))

    for n in range(numsims):
        reps = MakeReps(words, segment, None, **repArgs)
        
        reps /= numpy.reshape(numpy.sum(reps**2.0, 1)**0.5, (len(reps), 1)) # Normalize the representations
        
        if numToCompare == None:
            sim += numpy.dot(reps, numpy.transpose(reps))    # Computes the dot product of all representations
        else:
            for i in range(len(words)):
                temp_sims = numpy.array([numpy.dot(reps[i], reps[j]) for j in range(len(words))])
                sim[i] += numpy.sort(temp_sims)[(-2):(-2-numToCompare):-1]
    
    return sim / float(numsims)

def strCombos(toAdd, start=''):
    '''
    Generates all combinations of the elements in the list "toAdd".
    '''
    if len(toAdd) == 1: return [start+toAdd[0]]
    combos = []
    for i, item in enumerate(toAdd):
        combos.append(start + item)
        combos.extend(strCombos(toAdd[:i]+toAdd[(i+1):], start+item))
    return combos

def SegmentWords(words):
    '''
    Returns segmented forms of each of the words in the list "words".
    '''
    forms = [[] for word in words]
    
    for w, word in enumerate(words):
        forms[w].extend(segmentWord(word, words[:w] + words[(w+1):]))
    
    return forms
        
def segmentWord(word, lexicon):
    '''
    Searches the lexicon (a list of strings) for words that form part of the given
    word, and returns a list of all such possible decompositions of the word.
    '''
    segs = [[word]]
    if len(word) < 3: return segs
    
    for i in range(len(word)-2):
        for j in range(2, len(word)-i):
            if word[i:i+j] in lexicon:
                suffixes = segmentWord(word[i+j:], lexicon)
                for suffix in suffixes:
                    if i > 0: toAdd = [word[:i], word[i:i+j]]
                    else: toAdd = [word[i:i+j]]
                    toAdd.extend(suffix)
                    segs.append(toAdd)
    
    i = 0
    while i < len(segs):
        if segs[i] in segs[(i+1):]:
            del segs[i]
        else:
            i += 1
    
    return segs

def TopNSim(n = 100, words = 'elp_trimmed_words-freq.txt', numsims=1, filename=None, segment=False, thresholds = numpy.arange(0,1,.05), **repArgs):
    freq = []
    if type(words) == type(0):   # Then, words are the power set of the first 'words' letters
        words = strCombos([letter for letter in alphabet[:min(words, 26)]])
    if type(words) == type(''):
        word_list = []
        FIN = open(words, 'r')
        for line in FIN:
            line = line.strip().lower()
            line = line.split()
            word_list.append(line[0])
            if len(line) > 1:
                freq.append(float(line[1]))
            else:
                freq.append(1.0)
        FIN.close()
        if filename == None: filename = words+'sims.csv'
        words = word_list
    
    if len(freq) != len(words): freq = numpy.ones((len(words)))
    else: freq = numpy.array(freq)
    
    if n >= len(words): n = len(words) - 1
    
    aboveThreshold = numpy.zeros((len(words), len(thresholds)))
    numAboveThreshold = numpy.zeros((len(words), len(thresholds)))
    aboveThresholdVar = numpy.zeros((len(words), len(thresholds)))
    allMean = numpy.zeros((len(words)))
    allVar = numpy.zeros((len(words)))
    aboveThresholdF = numpy.zeros((len(words), len(thresholds)))
    allMeanF = numpy.zeros((len(words)))
    sorted_sim = numpy.zeros((len(words), n))
    sorted_freq = numpy.zeros((len(words), n))
    
    if len(words) < 1:
        closest_words = []
        sim = SimMatrix(words, numsims, segment=segment, **repArgs)
        sim -= 2.0*numpy.eye(len(sim))*sim
        for i in range(len(words)):
            notI = list(range(i))+list(range(i+1,len(words)))
            for t, threshold in enumerate(thresholds):
                aboveThreshold[i][t] = numpy.mean(sim[i, sim[i] > threshold])
                numAboveThreshold[i][t] = len(sim[i][sim[i] > threshold])
                aboveThresholdVar[i][t] = numpy.var(sim[i, sim[i] > threshold])
                aboveThresholdF[i][t] = numpy.dot(sim[i, sim[i] > threshold], freq[sim[i] > threshold] / numpy.sum(freq[sim[i]>threshold]))
            allMean[i] = numpy.mean(sim[i][notI])
            allVar[i] = numpy.var(sim[i][notI])
            allMeanF[i] = numpy.dot(sim[i][notI], freq[notI]) / numpy.sum(freq[notI])
            topSims = numpy.argsort(sim[i])[(len(sim[i])-1):(len(sim[i])-1-n):-1]
            sorted_sim[i] = sim[i][topSims]
            sorted_freq[i] = freq[topSims]
            closest_words.append(words[topSims[0]])
    else:
        FOUT = open(filename, 'w')
        FOUT.write('Word,ClosestWord,' + ','.join(['sim'+str(i) for i in range(sorted_sim.shape[1])]) + ',' + ','.join(['freq'+str(i) for i in range(sorted_sim.shape[1])]) + ',AllMean,AllVar,AllMeanF,' + ','.join(['NumAbove'+str(t) for t in thresholds]) + ',' + ','.join(['Above'+str(t) for t in thresholds]) + ',' + ','.join(['Above'+str(t)+'Var' for t in thresholds]) + ',' + ','.join(['Above'+str(t)+'F' for t in thresholds]) + '\n')
        reps = MakeReps(words, segment, None, **repArgs)
        aboveThreshold = numpy.zeros((len(thresholds)))
        aboveThresholdVar = numpy.zeros((len(thresholds)))
        numAboveThreshold = numpy.zeros((len(thresholds)))
        aboveThresholdF = numpy.zeros((len(thresholds)))
        eBins = numpy.arange(-1.0, 1.05, 0.05)
        for i in range(len(words)):
            notI = list(range(i))+list(range(i+1,len(words)))
            temp_sims = numpy.inner(reps[i], reps)
            temp_sims[i] = -1.0
            for t, threshold in enumerate(thresholds):
                aboveThreshold[t] = numpy.mean(temp_sims[temp_sims > threshold])
                numAboveThreshold[t] = len(temp_sims[temp_sims > threshold])
                aboveThresholdVar[t] = numpy.var(temp_sims[temp_sims > threshold])
                aboveThresholdF[t] = numpy.dot(temp_sims[temp_sims > threshold], freq[temp_sims > threshold]) / numpy.sum(freq[temp_sims > threshold])
            allMean = numpy.mean(temp_sims[notI])
            allVar = numpy.var(temp_sims[notI])
            allMeanF = numpy.dot(temp_sims[notI], freq[notI]) / numpy.sum(freq[notI])
            topSims = numpy.argsort(temp_sims)[(len(temp_sims)-1):(len(temp_sims)-1-n):-1]
            sorted_sim = temp_sims[topSims]
            sorted_freq = freq[topSims]
            
            FOUT.write(words[i]+','+words[topSims[0]]+','+ ','.join([str(s) for s in sorted_sim]) + ',' + ','.join([str(s) for s in sorted_freq])+','+str(allMean)+','+str(allVar)+','+str(allMeanF)+','+','.join([str(numAboveThreshold[t]) for t in range(len(thresholds))])+','+','.join([str(aboveThreshold[t]) for t in range(len(thresholds))])+','+','.join([str(aboveThresholdVar[t]) for t in range(len(thresholds))])+','+','.join([str(aboveThresholdF[t]) for t in range(len(thresholds))])+'\n')
            
            if i % 100 == 0: print(i, words[i])
        
        return
 
    
    if filename != None:
        FOUT = open(filename, 'w')
        FOUT.write('Word,ClosestWord,' + ','.join(['sim'+str(i) for i in range(sorted_sim.shape[1])]) + ',' + ','.join(['freq'+str(i) for i in range(sorted_sim.shape[1])]) + ',AllMean,AllMeanF,' + ','.join(['Above'+str(t) for t in thresholds]) + ',' + ','.join(['Above'+str(t)+'F' for t in thresholds]) + '\n')
        for i, word in enumerate(words):
            FOUT.write(word+','+closest_words[i]+','+ ','.join([str(s) for s in sorted_sim[i]]) + ',' + ','.join([str(s) for s in sorted_freq[i]])+','+str(allMean[i])+','+str(allMeanF[i])+','+','.join([str(aboveThreshold[i][t]) for t in range(len(thresholds))])+','+','.join([str(aboveThresholdF[i][t]) for t in range(len(thresholds))])+'\n')
        FOUT.close()
    return words, closest_words, sorted_sim, aboveThreshold, allMean, aboveThresholdF, allMeanF




def getPhonemes(word):
    READ = open("C:\\Users\\Stephen\\Downloads\\elpPhonsFull.csv", 'r')
    for line in READ:
        line = line.strip().lower()
        line = line.split(',')
        if (line[0] == word):
            ret = line[1].split()
            return ret
    return "NO PHONEMES FOUND"

def getPhonemesStress(word):
    if not (isinstance(word, str)):
        word = word[0][0]
    READ = open("C:\\Users\\Stephen\\Downloads\\elpPhonsFull.csv", 'r')
    for line in READ:
        line = line.strip().lower()
        line = line.split(',')
        if (line[0] == word):
            ret = line[2].split()
            return ret
    g2p = G2p()
    ret = g2p(word)
    ret = [s.lower() for s in ret]
    return g2p(ret)
#print(getPhonemesStress("walk"))
love = getPhonemesStress("love")
move = getPhonemesStress("move")
#print(love, move)
#print(dl("flaw", "spaw"))
#print(dl("flaw", "plaw"))
#print(dl(love, move))


def getTRNGramsPhon(seg, scale, spaces, minsize):
    '''
    Returns a list of n-grams from the string "seg", according to the "terminal
    relative" (TR) encoding procedure, i.e., any internal n-gram gets included
    both by itself and as part of a non-contiguous n-gram with n-grams at either
    end of "seg".  "Scale" is a list of ascending integers reflecting the sizes
    of the n-gram chunks.  "Spaces" indicates whether a space character should
    be used to mark gaps in non-contiguous n-grams.
    '''
    ngrams = []
    phons = getPhonemes(seg)
    for size in scale:
        if size > len(phons): break
        
        for i in range(len(phons)-size+1):
            if '_' in phons[i:i+size]: continue
            
            if size > minsize or len(phons) <= minsize: ngrams.append(phons[i:i+size])
            if phons[0] != '_':
                for fsize in scale:
                    if '_' in phons[:fsize] or fsize > i or (i==fsize and fsize+size in scale): break
                    ngrams.append(phons[:fsize] + (['_'] if spaces and i > fsize else ['']) + phons[i:i+size])
            if phons[-1] != '_':
                for fsize in scale:
                    if '_' in phons[-fsize:] or i + size > len(phons)-fsize or (i+size==len(phons)-fsize and fsize+size in scale): break
                    ngrams.append(phons[i:i+size] + (['_'] if spaces and i+size < len(phons)-fsize else ['']) + phons[-fsize:])
    
    for i in range(len(ngrams)):
        for j in range(len(ngrams[i])):
            while ('' in ngrams[i]):
                ngrams[i].remove('')
    return ngrams

def getOpenNGramsPhon(seg, scale, spaces, minsize):
    '''
    Returns a list of the open n-grams of the string "seg", with sizes specified
    by "scale", which should be a list of positive integers in ascending order.
    "Spaces" indicates whether a space character should be used to mark gaps in
    non-contiguous n-grams.
    '''
    ngrams = []
    phons = getPhonemes(seg)
    
    for size in scale:
        if size > len(phons): break
        
        for i in range(len(phons)):
            if i+size > len(phons): break
            if size > minsize or len(phons) <= minsize: ngrams.append(phons[i:i+size])
            if i+size == len(phons): continue
            for b in range(1, size):
                for e in range(1, len(phons)-i-size+1):
                    ngrams.append(phons[i:i+b]+(['_'] if spaces else [''])+phons[i+b+e:i+e+size])

    for i in range(len(ngrams)):
        for j in range(len(ngrams[i])):
            while ('' in ngrams[i]):
                ngrams[i].remove('')
    return ngrams
    

def getTRNGramsPhonStress(seg, scale, spaces, minsize):
    '''
    Returns a list of n-grams from the string "seg", according to the "terminal
    relative" (TR) encoding procedure, i.e., any internal n-gram gets included
    both by itself and as part of a non-contiguous n-gram with n-grams at either
    end of "seg".  "Scale" is a list of ascending integers reflecting the sizes
    of the n-gram chunks.  "Spaces" indicates whether a space character should
    be used to mark gaps in non-contiguous n-grams.
    '''
    ngrams = []
    phons = getPhonemesStress(seg)
    for size in scale:
        if size > len(phons): break
        
        for i in range(len(phons)-size+1):
            if '_' in phons[i:i+size]: continue
            
            if size > minsize or len(phons) <= minsize: ngrams.append(phons[i:i+size])
            if phons[0] != '_':
                for fsize in scale:
                    if '_' in phons[:fsize] or fsize > i or (i==fsize and fsize+size in scale): break
                    ngrams.append(phons[:fsize] + (['_'] if spaces and i > fsize else ['']) + phons[i:i+size])
            if phons[-1] != '_':
                for fsize in scale:
                    if '_' in phons[-fsize:] or i + size > len(phons)-fsize or (i+size==len(phons)-fsize and fsize+size in scale): break
                    ngrams.append(phons[i:i+size] + (['_'] if spaces and i+size < len(phons)-fsize else ['']) + phons[-fsize:])
    
    for i in range(len(ngrams)):
        for j in range(len(ngrams[i])):
            while ('' in ngrams[i]):
                ngrams[i].remove('')
    return ngrams

def getOpenNGramsPhonStress(seg, scale, spaces, minsize):
    '''
    Returns a list of the open n-grams of the string "seg", with sizes specified
    by "scale", which should be a list of positive integers in ascending order.
    "Spaces" indicates whether a space character should be used to mark gaps in
    non-contiguous n-grams.
    '''
    ngrams = []
    if len(seg) == 1:
        phons = getPhonemesStress(seg)
    else:
        phons = [s.lower() for s in seg]
    
    
    for size in scale:
        if size > len(phons): break
        
        for i in range(len(phons)):
            if i+size > len(phons): break
            if size > minsize or len(phons) <= minsize: ngrams.append(phons[i:i+size])
            if i+size == len(phons): continue
            for b in range(1, size):
                for e in range(1, len(phons)-i-size+1):
                    ngrams.append(phons[i:i+b]+(['_'] if spaces else [''])+phons[i+b+e:i+e+size])

    for i in range(len(ngrams)):
        for j in range(len(ngrams[i])):
            while ('' in ngrams[i]):
                ngrams[i].remove('')
    return ngrams


def getTRNGramsIPA(seg, scale, spaces, minsize):
    '''
    Returns a list of n-grams from the string "seg", according to the "terminal
    relative" (TR) encoding procedure, i.e., any internal n-gram gets included
    both by itself and as part of a non-contiguous n-gram with n-grams at either
    end of "seg".  "Scale" is a list of ascending integers reflecting the sizes
    of the n-gram chunks.  "Spaces" indicates whether a space character should
    be used to mark gaps in non-contiguous n-grams.
    '''
    ngrams = []
    phons = getPhonemes(seg)
    for size in scale:
        if size > len(phons): break
        
        for i in range(len(phons)-size+1):
            if '_' in phons[i:i+size]: continue
            
            if size > minsize or len(phons) <= minsize: ngrams.append(phons[i:i+size])
            if phons[0] != '_':
                for fsize in scale:
                    if '_' in phons[:fsize] or fsize > i or (i==fsize and fsize+size in scale): break
                    ngrams.append(phons[:fsize] + (['_'] if spaces and i > fsize else ['']) + phons[i:i+size])
            if phons[-1] != '_':
                for fsize in scale:
                    if '_' in phons[-fsize:] or i + size > len(phons)-fsize or (i+size==len(phons)-fsize and fsize+size in scale): break
                    ngrams.append(phons[i:i+size] + (['_'] if spaces and i+size < len(phons)-fsize else ['']) + phons[-fsize:])
    
    for i in range(len(ngrams)):
        for j in range(len(ngrams[i])):
            while ('' in ngrams[i]):
                ngrams[i].remove('')
    
    return ngrams

def getOpenNGramsIPA(seg, scale, spaces, minsize):
    '''
    Returns a list of the open n-grams of the string "seg", with sizes specified
    by "scale", which should be a list of positive integers in ascending order.
    "Spaces" indicates whether a space character should be used to mark gaps in
    non-contiguous n-grams.
    '''
    ngrams = []
    phons = getPhonemes(seg)
    
    for size in scale:
        if size > len(phons): break
        
        for i in range(len(phons)):
            if i+size > len(phons): break
            if size > minsize or len(phons) <= minsize: ngrams.append(phons[i:i+size])
            if i+size == len(phons): continue
            for b in range(1, size):
                for e in range(1, len(phons)-i-size+1):
                    ngrams.append(phons[i:i+b]+(['_'] if spaces else [''])+phons[i+b+e:i+e+size])

    for i in range(len(ngrams)):
        for j in range(len(ngrams[i])):
            while ('' in ngrams[i]):
                ngrams[i].remove('')
    return ngrams

    






class PhonWordRep:
    '''
    Spawns objects capable of generating holographic word-form representations
    in both the visual and auditory modalities. Auditory representations depend
    on the given word being present in the CMU Pronouncing Dictionary.
    '''
    
    def __init__(self, d = 1024, ngramType = 'tr', vis_scale=[1,2], spaces = True, minsize = 0, bindOp = 'convolution', seed = None):
        '''
        Creates a new holographic word-form representation generator.
        d - dimensionality of the representations (typically quite large)
        ngramType - how words are to be decomposed into n-grams; either 'tr'
            (terminal relative) or 'open'.
        vis_scale - a list of the size of successive chunks in visual word
            representations (in # of letters)
        spaces - whether or not to use a space character to mark gaps in non-
            contiguous n-grams
        bindOp - one of "convolution", "addition", or "bsc" (or just the initial
            letter of those) indicating the method by which n-grams should be
            bound together
        seed - integer seed for the numpy random number generator (optional)
        '''
        
        numpy.random.seed(seed)
        self.d = d
        self.vis_scale = sorted(vis_scale)
        self.spaces = spaces
        # Set the n-gram extraction method
        if ngramType.lower().startswith('t'):
            self.getNGramsPhon = lambda s: getTRNGramsPhon(s, self.vis_scale, self.spaces, minsize)
        elif ngramType.lower().startswith('o'):
            self.getNGramsPhon = lambda s: getOpenNGramsPhon(s, self.vis_scale, self.spaces, minsize)
        else:
            raise ArgumentError('Invalid n-gram type!')
        
        bindOp = bindOp.lower()
        
        if bindOp.startswith('c') or bindOp.startswith('a'):
            # Create random vectors representing individual letters, the "atoms"
            # of the word representations.
            self.phons = dict(list(zip(phonemes, [normalize(numpy.random.randn(d) * d**-0.5) for p in phonemes])))
            self.chunk = lambda l: normalize(reduce(lambda a,b: a+b, l, numpy.zeros((d))))    #Chunking is done by superposition (vector addition)
            
            
            if bindOp.startswith('c'):
                # Permutation operators that scramble the vectors in a
                # convolution operation; this makes the operation non-commutative and
                # thus allows it to encode order.
                self.place1 = numpy.random.permutation(d)
                self.place2 = numpy.random.permutation(d)
                
                self.invplace1 = numpy.zeros((d), dtype='int')
                self.invplace2 = numpy.zeros((d), dtype='int')
                
                for i in range(d):
                    self.invplace1[self.place1[i]] = i
                    self.invplace2[self.place2[i]] = i


                self.bind = lambda l: convBind(self.place1, self.place2, l)
            else:
                # Here, to make binding non-commutative, each operand is first
                # convolved with a place vector, and the two results are added.
                # (Results in "fuzzy position" coding)
                self.place1 = normalize(numpy.random.randn(d) * d**-0.5)
                self.place2 = normalize(numpy.random.randn(d) * d**-0.5)
                
                self.bind = lambda l: addBind(self.place1, self.place2, l)
        elif bindOp.startswith('b'):
            # Create random vectors representing individual letters, the "atoms"
            # of the word representations.  These are binary vectors taking the
            # values -1 or 1.
            self.phons = dict(list(zip(phonemes, [(numpy.random.rand(d) < 0.5) * 2.0 - 1.0 for p in phonemes])))
            
            self.place1 = (numpy.random.rand(d) < 0.5) * 2.0 - 1.0
            self.place2 = (numpy.random.rand(d) < 0.5) * 2.0 - 1.0
            
            self.chunk = lambda l: maj(l, 0.5)
            self.bind = lambda l: bscBind(self.place1, self.place2, l, 0.5)
        elif bindOp.startswith('sl'):
            self.getNGramsPhon = lambda s: [s]
            self.phons = dict(list(zip(phonemes, [normalize(numpy.random.randn(d) * d**-0.5) for p in phonemes])))
            #print(self.phons)
            self.place = normalize(numpy.random.randn(d) * d**-0.5)
            
            self.chunk = lambda l: normalize(reduce(lambda a,b: a+b, l, numpy.zeros((d))))    #Chunking is done by superposition (vector addition)
            self.bind = lambda l: slotBind(self.place, l)
        elif bindOp.startswith('sp'):
            self.getNGramsPhon = lambda s: [s]
            self.phons = dict(list(zip(phonemes, [normalize(numpy.random.randn(d) * d**-0.5) for p in phonemes])))
            self.place = normalize(numpy.random.randn(d) * d**-0.5)
            
            self.chunk = lambda l: normalize(reduce(lambda a,b: a+b, l, numpy.zeros((d))))    #Chunking is done by superposition (vector addition)
            self.bind = lambda l: spatialBind(self.place, l)
        else:
            raise ArgumentError('Invalid binding operator specification!')
    
    def make_repPhon(self, word):
        '''
        Returns a holographic representation of the given word. The word may be
        given as just a string, or as a list of lists, where each sublist is a
        way of segmenting the word, e.g., "homework" vs. [['homework'], ['home',
        'work']].
        '''
        if type(word) == type(''): word = [ [word] ]
        # Create a visual word form representation based on the phonemes present
        # in the word.
        formReps = []
        for form in word:
            segReps = []
            
            for seg in form:
                seg = seg.strip().lower()
                ngrams = self.getNGramsPhon(seg)
                #print(ngrams)
                segRep = self.chunk([self.bind([self.phons[ph] for ph in ngram]) for ngram in ngrams])
                #print(self.phons)
                segReps.append(segRep)
            if len(segReps) > 1:
                segChunks = []
                for size in range(1,len(segReps)):
                    for i in range(len(segReps)-size+1):
                        segChunks.append(self.bind(segReps[i:(i+size)]))
            else:
                segChunks = [segReps[0]]
            
            formReps.append(self.chunk(segChunks))
        
        return self.chunk(formReps)


def MakeRepsPhon(words, segment = False, filename = None, **repArgs):
    '''
    Makes representations for the words in "words", which can be a list of strings
    or a string giving a filename.
    segment - Whether or not to find ways of segmenting each word into other,
        shorter words (e.g., "homework" = "home" + "work")
    filename - A filename to which the representations can be written in CSV format
    repArgs - named arguments to be passed to HoloWordRep
    '''
    if type(words) == type(''):
        word_list = []
        FIN = open(words, 'r')
        for line in FIN:
            line = line.strip().lower()
            line = line.split()
            word_list.append(line[0])
        FIN.close()
        words = word_list
        
    h = PhonWordRep(**repArgs)
    
    if segment:
        forms = SegmentWords(words)
        reps = numpy.array([h.make_repPhon(form) for form in forms])
    else:
        reps = numpy.array([h.make_repPhon(word) for word in words])            # Make representations for all words in the list
    
    if filename != None:
        FOUT = open(filename, 'w')
        for i, word in enumerate(words):
            FOUT.write(word + ',' + ','.join([str(x) for x in reps[i]]) + '\n')
        FOUT.close()
    
    return reps

def TopNSimPhon(n = 100, words = 'elp_trimmed_words-freq.txt', numsims=1, filename=None, segment=False, thresholds = numpy.arange(0,1,.05), stress = False, **repArgs):
    freq = []
    if type(words) == type(0):   # Then, words are the power set of the first 'words' letters
        words = strCombos([letter for letter in phonemes[:min(words, 39)]])
    if type(words) == type(''):
        word_list = []
        FIN = open(words, 'r')
        for line in FIN:
            line = line.strip().lower()
            line = line.split()
            word_list.append(line[0])
            if len(line) > 1:
                freq.append(float(line[1]))
            else:
                freq.append(1.0)
        FIN.close()
        if filename == None: filename = words+'sims.csv'
        words = word_list
    
    if len(freq) != len(words): freq = numpy.ones((len(words)))
    else: freq = numpy.array(freq)
    
    if n >= len(words): n = len(words) - 1
    
    aboveThreshold = numpy.zeros((len(words), len(thresholds)))
    numAboveThreshold = numpy.zeros((len(words), len(thresholds)))
    aboveThresholdVar = numpy.zeros((len(words), len(thresholds)))
    allMean = numpy.zeros((len(words)))
    allVar = numpy.zeros((len(words)))
    aboveThresholdF = numpy.zeros((len(words), len(thresholds)))
    allMeanF = numpy.zeros((len(words)))
    sorted_sim = numpy.zeros((len(words), n))
    sorted_freq = numpy.zeros((len(words), n))
    
    if len(words) < 1:
        closest_words = []
        sim = SimMatrix(words, numsims, segment=segment, **repArgs)
        sim -= 2.0*numpy.eye(len(sim))*sim
        for i in range(len(words)):
            notI = list(range(i))+list(range(i+1,len(words)))
            for t, threshold in enumerate(thresholds):
                aboveThreshold[i][t] = numpy.mean(sim[i, sim[i] > threshold])
                numAboveThreshold[i][t] = len(sim[i][sim[i] > threshold])
                aboveThresholdVar[i][t] = numpy.var(sim[i, sim[i] > threshold])
                aboveThresholdF[i][t] = numpy.dot(sim[i, sim[i] > threshold], freq[sim[i] > threshold] / numpy.sum(freq[sim[i]>threshold]))
            allMean[i] = numpy.mean(sim[i][notI])
            allVar[i] = numpy.var(sim[i][notI])
            allMeanF[i] = numpy.dot(sim[i][notI], freq[notI]) / numpy.sum(freq[notI])
            topSims = numpy.argsort(sim[i])[(len(sim[i])-1):(len(sim[i])-1-n):-1]
            sorted_sim[i] = sim[i][topSims]
            sorted_freq[i] = freq[topSims]
            closest_words.append(words[topSims[0]])
    else:
        FOUT = open(filename, 'w')
        FOUT.write('Word,ClosestWord,' + ','.join(['sim'+str(i) for i in range(sorted_sim.shape[1])]) + ',' + ','.join(['freq'+str(i) for i in range(sorted_sim.shape[1])]) + ',AllMean,AllVar,AllMeanF,' + ','.join(['NumAbove'+str(t) for t in thresholds]) + ',' + ','.join(['Above'+str(t) for t in thresholds]) + ',' + ','.join(['Above'+str(t)+'Var' for t in thresholds]) + ',' + ','.join(['Above'+str(t)+'F' for t in thresholds]) + '\n')
        reps = MakeRepsPhon(words, segment, None, stress, **repArgs)
        aboveThreshold = numpy.zeros((len(thresholds)))
        aboveThresholdVar = numpy.zeros((len(thresholds)))
        numAboveThreshold = numpy.zeros((len(thresholds)))
        aboveThresholdF = numpy.zeros((len(thresholds)))
        eBins = numpy.arange(-1.0, 1.05, 0.05)
        for i in range(len(words)):
            notI = list(range(i))+list(range(i+1,len(words)))
            temp_sims = numpy.inner(reps[i], reps)
            temp_sims[i] = -1.0
            for t, threshold in enumerate(thresholds):
                aboveThreshold[t] = numpy.mean(temp_sims[temp_sims > threshold])
                numAboveThreshold[t] = len(temp_sims[temp_sims > threshold])
                aboveThresholdVar[t] = numpy.var(temp_sims[temp_sims > threshold])
                aboveThresholdF[t] = numpy.dot(temp_sims[temp_sims > threshold], freq[temp_sims > threshold]) / numpy.sum(freq[temp_sims > threshold])
            allMean = numpy.mean(temp_sims[notI])
            allVar = numpy.var(temp_sims[notI])
            allMeanF = numpy.dot(temp_sims[notI], freq[notI]) / numpy.sum(freq[notI])
            topSims = numpy.argsort(temp_sims)[(len(temp_sims)-1):(len(temp_sims)-1-n):-1]
            sorted_sim = temp_sims[topSims]
            sorted_freq = freq[topSims]
            
            FOUT.write(words[i]+','+words[topSims[0]]+','+words[topSims[1]]+','+words[topSims[2]]+','+words[topSims[3]]+','+words[topSims[4]]+','+words[topSims[5]]+','+words[topSims[6]]+','+words[topSims[7]]+','+words[topSims[8]]+','+words[topSims[9]]+','+ ','.join([str(s) for s in sorted_sim]) + ',' + ','.join([str(s) for s in sorted_freq])+','+str(allMean)+','+str(allVar)+','+str(allMeanF)+','+','.join([str(numAboveThreshold[t]) for t in range(len(thresholds))])+','+','.join([str(aboveThreshold[t]) for t in range(len(thresholds))])+','+','.join([str(aboveThresholdVar[t]) for t in range(len(thresholds))])+','+','.join([str(aboveThresholdF[t]) for t in range(len(thresholds))])+'\n')
        
        return
        
    
    if filename != None:
        FOUT = open(filename, 'w')
        FOUT.write('Word,ClosestWord,' + ','.join(['sim'+str(i) for i in range(sorted_sim.shape[1])]) + ',' + ','.join(['freq'+str(i) for i in range(sorted_sim.shape[1])]) + ',AllMean,AllMeanF,' + ','.join(['Above'+str(t) for t in thresholds]) + ',' + ','.join(['Above'+str(t)+'F' for t in thresholds]) + '\n')
        for i, word in enumerate(words):
            FOUT.write(word+','+closest_words[i]+','+ ','.join([str(s) for s in sorted_sim[i]]) + ',' + ','.join([str(s) for s in sorted_freq[i]])+','+str(allMean[i])+','+str(allMeanF[i])+','+','.join([str(aboveThreshold[i][t]) for t in range(len(thresholds))])+','+','.join([str(aboveThresholdF[i][t]) for t in range(len(thresholds))])+'\n')
        FOUT.close()
    return words, closest_words, sorted_sim, aboveThreshold, allMean, aboveThresholdF, allMeanF


def cosSim(a, b):
    #the closer the cosine similarity is to 1 or -1, the more similar the vectors are, and the closer it is to 0, the more disimilar they are
    return np.dot(a,b)/(norm(b)*norm(b))

def Average(lst): 
    return sum(lst) / len(lst) 



class PhonWordRepStress:
    '''
    Spawns objects capable of generating holographic word-form representations
    in both the visual and auditory modalities. Auditory representations depend
    on the given word being present in the CMU Pronouncing Dictionary.
    '''
    
    def __init__(self, d = 1024, ngramType = 'tr', vis_scale=[1,2], spaces = True, minsize = 0, bindOp = 'convolution', seed = None):
        '''
        Creates a new holographic word-form representation generator.
        d - dimensionality of the representations (typically quite large)
        ngramType - how words are to be decomposed into n-grams; either 'tr'
            (terminal relative) or 'open'.
        vis_scale - a list of the size of successive chunks in visual word
            representations (in # of letters)
        spaces - whether or not to use a space character to mark gaps in non-
            contiguous n-grams
        bindOp - one of "convolution", "addition", or "bsc" (or just the initial
            letter of those) indicating the method by which n-grams should be
            bound together
        seed - integer seed for the numpy random number generator (optional)
        '''
        
        numpy.random.seed(seed)
        self.d = d
        self.vis_scale = sorted(vis_scale)
        self.spaces = spaces
        # Set the n-gram extraction method
        if ngramType.lower().startswith('t'):
            self.getNGramsPhonStress = lambda s: getTRNGramsPhonStress(s, self.vis_scale, self.spaces, minsize)
        elif ngramType.lower().startswith('o'):
            self.getNGramsPhonStress = lambda s: getOpenNGramsPhonStress(s, self.vis_scale, self.spaces, minsize)
        else:
            raise ArgumentError('Invalid n-gram type!')
        
        bindOp = bindOp.lower()
        
        if bindOp.startswith('c') or bindOp.startswith('a'):
            # Create random vectors representing individual letters, the "atoms"
            # of the word representations.
            self.phons = dict(list(zip(phonemesStress, [normalize(numpy.random.randn(d) * d**-0.5) for p in phonemesStress])))
            '''
            for key in self.phons:
                if key.upper() in stressVowels:
                    if '0' in key:
                        self.phons[key] = self.phons[key] * 0.2
                    if '2' in key:
                        self.phons[key] = self.phons[key] * 0.75
            '''
            self.chunk = lambda l: normalize(reduce(lambda a,b: a+b, l, numpy.zeros((d))))    #Chunking is done by superposition (vector addition)
            
            
            if bindOp.startswith('c'):
                # Permutation operators that scramble the vectors in a
                # convolution operation; this makes the operation non-commutative and
                # thus allows it to encode order.
                self.place1 = numpy.random.permutation(d)
                self.place2 = numpy.random.permutation(d)
                
                self.invplace1 = numpy.zeros((d), dtype='int')
                self.invplace2 = numpy.zeros((d), dtype='int')
                
                for i in range(d):
                    self.invplace1[self.place1[i]] = i
                    self.invplace2[self.place2[i]] = i


                self.bind = lambda l: convBind(self.place1, self.place2, l)
            else:
                # Here, to make binding non-commutative, each operand is first
                # convolved with a place vector, and the two results are added.
                # (Results in "fuzzy position" coding)
                self.place1 = normalize(numpy.random.randn(d) * d**-0.5)
                self.place2 = normalize(numpy.random.randn(d) * d**-0.5)
                
                self.bind = lambda l: addBind(self.place1, self.place2, l)
        elif bindOp.startswith('b'):
            # Create random vectors representing individual letters, the "atoms"
            # of the word representations.  These are binary vectors taking the
            # values -1 or 1.
            self.phons = dict(list(zip(phonemesStress, [(numpy.random.rand(d) < 0.5) * 2.0 - 1.0 for p in phonemesStress])))
            
            self.place1 = (numpy.random.rand(d) < 0.5) * 2.0 - 1.0
            self.place2 = (numpy.random.rand(d) < 0.5) * 2.0 - 1.0
            
            self.chunk = lambda l: maj(l, 0.5)
            self.bind = lambda l: bscBind(self.place1, self.place2, l, 0.5)
        elif bindOp.startswith('sl'):
            self.getNGramsPhon = lambda s: [s]
            self.phons = dict(list(zip(alphabet, [normalize(numpy.random.randn(d) * d**-0.5) for letter in alphabet])))
            self.place = normalize(numpy.random.randn(d) * d**-0.5)
            
            self.chunk = lambda l: normalize(reduce(lambda a,b: a+b, l, numpy.zeros((d))))    #Chunking is done by superposition (vector addition)
            self.bind = lambda l: slotBind(self.place, l)
        elif bindOp.startswith('sp'):
            self.getNGramsPhon = lambda s: [s]
            self.phons = dict(list(zip(alphabet, [normalize(numpy.random.randn(d) * d**-0.5) for letter in alphabet])))
            self.place = normalize(numpy.random.randn(d) * d**-0.5)
            
            self.chunk = lambda l: normalize(reduce(lambda a,b: a+b, l, numpy.zeros((d))))    #Chunking is done by superposition (vector addition)
            self.bind = lambda l: spatialBind(self.place, l)
        else:
            raise ArgumentError('Invalid binding operator specification!')
    
    def make_repPhonStress(self, word):
        '''
        Returns a holographic representation of the given word. The word may be
        given as just a string, or as a list of lists, where each sublist is a
        way of segmenting the word, e.g., "homework" vs. [['homework'], ['home',
        'work']].
        '''
        if type(word) == type(''): word = [ [word] ]
        # Create a visual word form representation based on the phonemes present
        # in the word.
        formReps = []
        for form in word:
            segReps = []
            
            for seg in form:
                seg = seg.strip().lower()
                if isinstance(word, str):
                    ngrams = self.getNGramsPhonStress(seg)
                else:
                    ngrams = self.getNGramsPhonStress(word)
                segRep = self.chunk([self.bind([self.phons[ph] for ph in ngram]) for ngram in ngrams])
                segReps.append(segRep)
            if len(segReps) > 1:
                segChunks = []
                for size in range(1,len(segReps)):
                    for i in range(len(segReps)-size+1):
                        segChunks.append(self.bind(segReps[i:(i+size)]))
            else:
                segChunks = [segReps[0]]
            
            formReps.append(self.chunk(segChunks))
        
        return self.chunk(formReps)
    


class IPAWordRep:
    '''
    Spawns objects capable of generating holographic word-form representations
    in both the visual and auditory modalities. Auditory representations depend
    on the given word being present in the CMU Pronouncing Dictionary.
    '''
    
    def __init__(self, d = 1024, ngramType = 'tr', vis_scale=[1,2], spaces = True, minsize = 0, bindOp = 'convolution', seed = None, first = True):
        '''
        Creates a new holographic word-form representation generator.
        d - dimensionality of the representations (typically quite large)
        ngramType - how words are to be decomposed into n-grams; either 'tr'
            (terminal relative) or 'open'.
        vis_scale - a list of the size of successive chunks in visual word
            representations (in # of letters)
        spaces - whether or not to use a space character to mark gaps in non-
            contiguous n-grams
        bindOp - one of "convolution", "addition", or "bsc" (or just the initial
            letter of those) indicating the method by which n-grams should be
            bound together
        seed - integer seed for the numpy random number generator (optional)
        '''
        
        numpy.random.seed(seed)
        self.d = d
        self.vis_scale = sorted(vis_scale)
        self.spaces = spaces
        # Set the n-gram extraction method
        if ngramType.lower().startswith('t'):
            self.getNGramsPhon = lambda s: getTRNGramsIPA(s, self.vis_scale, self.spaces, minsize)
        elif ngramType.lower().startswith('o'):
            self.getNGramsPhon = lambda s: getOpenNGramsIPA(s, self.vis_scale, self.spaces, minsize)
        else:
            raise ArgumentError('Invalid n-gram type!')
        
        bindOp = bindOp.lower()
        
        if bindOp.startswith('c') or bindOp.startswith('a'):
            # Create random vectors representing individual letters, the "atoms"
            # of the word representations.
            
            self.phons = dict(list(zip(IPAPhons, [normalize(numpy.random.randn(d) * d**-0.5) for p in IPAPhons])))
            if (first):
                reps = MakeRepsPhon(phonemes, False, None, True)
                for i in range(len(phonemes)):
                    self.phons[phonemes[i]] = reps[i]
            
            self.chunk = lambda l: normalize(reduce(lambda a,b: a+b, l, numpy.zeros((d))))    #Chunking is done by superposition (vector addition)
            
            
            if bindOp.startswith('c'):
                # Permutation operators that scramble the vectors in a
                # convolution operation; this makes the operation non-commutative and
                # thus allows it to encode order.
                self.place1 = numpy.random.permutation(d)
                self.place2 = numpy.random.permutation(d)
                
                self.invplace1 = numpy.zeros((d), dtype='int')
                self.invplace2 = numpy.zeros((d), dtype='int')
                
                for i in range(d):
                    self.invplace1[self.place1[i]] = i
                    self.invplace2[self.place2[i]] = i


                self.bind = lambda l: convBind(self.place1, self.place2, l)
            else:
                # Here, to make binding non-commutative, each operand is first
                # convolved with a place vector, and the two results are added.
                # (Results in "fuzzy position" coding)
                self.place1 = normalize(numpy.random.randn(d) * d**-0.5)
                self.place2 = normalize(numpy.random.randn(d) * d**-0.5)
                
                self.bind = lambda l: addBind(self.place1, self.place2, l)
        elif bindOp.startswith('b'):
            # Create random vectors representing individual letters, the "atoms"
            # of the word representations.  These are binary vectors taking the
            # values -1 or 1.
            self.phons = dict(list(zip(IPAPhons, [(numpy.random.rand(d) < 0.5) * 2.0 - 1.0 for p in IPAPhons])))
            if (first):
                reps = MakeRepsPhon(phonemes, False, None, True)
                for i in range(len(phonemes)):
                    self.phons[phonemes[i]] = reps[i]

            self.place1 = (numpy.random.rand(d) < 0.5) * 2.0 - 1.0
            self.place2 = (numpy.random.rand(d) < 0.5) * 2.0 - 1.0
            
            self.chunk = lambda l: maj(l, 0.5)
            self.bind = lambda l: bscBind(self.place1, self.place2, l, 0.5)
        elif bindOp.startswith('sl'):
            self.getNGramsPhon = lambda s: [s]
            self.phons = dict(list(zip(alphabet, [normalize(numpy.random.randn(d) * d**-0.5) for letter in alphabet])))
            self.place = normalize(numpy.random.randn(d) * d**-0.5)
            
            self.chunk = lambda l: normalize(reduce(lambda a,b: a+b, l, numpy.zeros((d))))    #Chunking is done by superposition (vector addition)
            self.bind = lambda l: slotBind(self.place, l)
        elif bindOp.startswith('sp'):
            self.getNGramsPhon = lambda s: [s]
            self.phons = dict(list(zip(alphabet, [normalize(numpy.random.randn(d) * d**-0.5) for letter in alphabet])))
            self.place = normalize(numpy.random.randn(d) * d**-0.5)
            
            self.chunk = lambda l: normalize(reduce(lambda a,b: a+b, l, numpy.zeros((d))))    #Chunking is done by superposition (vector addition)
            self.bind = lambda l: spatialBind(self.place, l)
        else:
            raise ArgumentError('Invalid binding operator specification!')
        
    def make_repIPA(self, word):
        '''
        Returns a holographic representation of the given word. The word may be
        given as just a string, or as a list of lists, where each sublist is a
        way of segmenting the word, e.g., "homework" vs. [['homework'], ['home',
        'work']].
        '''
        if type(word) == type(''): word = [ [word] ]
        # Create a visual word form representation based on the phonemes present
        # in the word.
        formReps = []
        for form in word:
            segReps = []
            
            for seg in form:
                seg = seg.strip().lower()
                ngrams = []
                for ph in phone_feature_map[seg.upper()]:
                    ngrams.append([ph])
                ngrams.append(phone_feature_map[seg.upper()])
                segRep = self.chunk([self.bind([self.phons[ph] for ph in ngram]) for ngram in ngrams])
                segReps.append(segRep)
            if len(segReps) > 1:
                segChunks = []
                for size in range(1,len(segReps)):
                    for i in range(len(segReps)-size+1):
                        segChunks.append(self.bind(segReps[i:(i+size)]))
            else:
                segChunks = [segReps[0]]
            
            formReps.append(self.chunk(segChunks))
        
        return self.chunk(formReps)
    
    def make_repPhon(self, word):
        '''
        Returns a holographic representation of the given word. The word may be
        given as just a string, or as a list of lists, where each sublist is a
        way of segmenting the word, e.g., "homework" vs. [['homework'], ['home',
        'work']].
        '''
        if type(word) == type(''): word = [ [word] ]
        # Create a visual word form representation based on the phonemes present
        # in the word.
        formReps = []
        for form in word:
            segReps = []
            
            for seg in form:
                seg = seg.strip().lower()
                ngrams = self.getNGramsPhon(seg)
                for thing in IPAPhons:
                    if (phonemes.count(thing) == 0 and thing in self.phons):
                        del self.phons[thing]
                segRep = self.chunk([self.bind([self.phons[ph] for ph in ngram]) for ngram in ngrams])
                segReps.append(segRep)
            if len(segReps) > 1:
                segChunks = []
                for size in range(1,len(segReps)):
                    for i in range(len(segReps)-size+1):
                        segChunks.append(self.bind(segReps[i:(i+size)]))
            else:
                segChunks = [segReps[0]]
            
            formReps.append(self.chunk(segChunks))
        
        return self.chunk(formReps)


def MakeRepsPhon(words, segment = False, filename = None, IPA = False, **repArgs):
    '''
    Makes representations for the words in "words", which can be a list of strings
    or a string giving a filename.
    segment - Whether or not to find ways of segmenting each word into other,
        shorter words (e.g., "homework" = "home" + "work")
    filename - A filename to which the representations can be written in CSV format
    repArgs - named arguments to be passed to HoloWordRep
    '''
    if type(words) == type(''):
        word_list = []
        FIN = open(words, 'r')
        for line in FIN:
            line = line.strip().lower()
            line = line.split()
            word_list.append(line[0])
        FIN.close()
        words = word_list
        
    b = IPAWordRep(first = False, **repArgs)
    
    if segment:
        forms = SegmentWords(words)
        if (IPA):
            reps = numpy.array([b.make_repIPA(form) for form in forms])
        else:
            reps = numpy.array([b.make_repPhon(form) for form in forms])
    else:
        if (IPA):
            reps = numpy.array([b.make_repIPA(word) for word in words])
        else:
            reps = numpy.array([b.make_repPhon(word) for word in words])             # Make representations for all words in the list
    
    if filename != None:
        FOUT = open(filename, 'w')
        for i, word in enumerate(words):
            FOUT.write(word + ',' + ','.join([str(x) for x in reps[i]]) + '\n')
        FOUT.close()
    
    return reps



