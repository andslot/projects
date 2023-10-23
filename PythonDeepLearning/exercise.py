import math
#Exercise 1
ENGLISH_FREQ = (0.0817,0.0149,0.0278,0.0425,0.1270,0.0223,0.0202,0.0609,0.0697,0.0015,0.0077,0.0403,0.0241,0.0675,0.0751,0.0193,0.0010,0.0599,0.0633,0.0906,0.0276,0.0098,0.0236,0.0015,0.0197,0.0007)
entropy = 0
entropies = [0]*26
for index,elem in enumerate(ENGLISH_FREQ):
    elem_entropy = elem*math.log2(1/elem)+(1-elem)*math.log2(1/(1-elem))
    entropies[index]=(chr(index+ord('A')),elem_entropy)
    entropy += elem_entropy
print(entropy*(1/26))
#Exercise 2
print(sorted(entropies, key=lambda x: x[1])[::-1])

#Exercise 3
#Yessir

#Exercise 4
#No it depends entirely on the frequency distribution