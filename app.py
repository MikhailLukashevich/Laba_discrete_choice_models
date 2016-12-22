import json
from hmmlearn import hmm
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

with open('corpus.json') as f:
    corpus = json.load(f)

# init model for phoneme 'a'
model_a = hmm.GaussianHMM(n_components=3, covariance_type='diag')
# init model for phoneme 'o'
model_o = hmm.GaussianHMM(n_components=3, covariance_type='diag')

#leraning phoneme 'a', lengths leraning set l_a = len(corpus['a'])*0.8 = 179
lengths_a = []
concat_a = []
for i in range(int(len(corpus['a'])*0.8)):
    lengths_a.append(len(corpus['a'][i]))
    concat_a += corpus['a'][i]

concat_a = np.asarray(concat_a)
print('----------DATA PHONEME "a"----------')
# print (concat_a)
# print (lengths_a)
print ("lengths all set ", len(corpus['a']))
print ("lengths leraning set ", int(len(corpus['a'])*0.8))
print ("lengths test set ", (len(corpus['a']) - int(len(corpus['a'])*0.8)))
print ("list with lengths appropriate initial sequence ", len(lengths_a))
print ("long sequence (concatenate) ", len(concat_a))

# create model for phoneme 'a'
model_a.fit(concat_a, lengths_a)
#print (model_a)


#leraning phoneme 'o', lengths leraning set l_0 = len(corpus['o'])*0.8 = 41
lengths_o = []
concat_o = []
for i in range(int(len(corpus['o'])*0.8)):
    lengths_o.append(len(corpus['o'][i]))
    concat_o += corpus['o'][i]

concat_o = np.asarray(concat_o)
print('----------DATA PHONEME "o"----------')
# print (concat_o)
# print (lengths_o)
print ("lengths all set ", len(corpus['o']))
print ("lengths leraning set ", int(len(corpus['o'])*0.8))
print ("lengths test set ", (len(corpus['o']) - int(len(corpus['o'])*0.8)))
print ("list with lengths appropriate initial sequence ", len(lengths_o))
print ("long sequence (concatenate) ", len(concat_o))

model_o.fit(concat_o, lengths_o)
#print (model_o)


# recognition phoneme 'a', using model_a and model_o, lengths set = all lengths set for phoneme 'a'
a_true = 0
a_false = 0
for i in range(len(corpus['a'])):
    xa = corpus['a'][i]
    if model_a.score(xa) > model_o.score(xa):
        a_true += 1
    else:
        a_false += 1

print('----------RECOGNITION PHONEME "a"----------')
print("a_true", a_true)
print("a_false", a_false)

acc_a = float(a_true) / len(corpus['a'])
print ("recognition accuracy for phoneme 'a', acc_a= ",acc_a)

# recognition phoneme 'o', using model_a and model_o, lengths set = all lengths set for phoneme 'o'
o_true = 0
o_false = 0
for i in range(len(corpus['o'])):
    xo = corpus['o'][i]
    if model_o.score(xo) > model_a.score(xo):
        o_true += 1
    else:
        o_false += 1

print('----------RECOGNITION PHONEME "o"----------')
print("o_true", o_true)
print("o_false", o_false)

acc_o = float(o_true) / len(corpus['o'])
print ("recognition accuracy for phoneme 'o', acc_o= ",acc_o)
