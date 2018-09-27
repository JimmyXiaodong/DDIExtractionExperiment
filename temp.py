import os
import numpy as np

BASE_DIR = ''
GLOVE_DIR = BASE_DIR + 'glove.840B/'

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.840B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


word1 = 'concomitant'
word2 = 'drug'
word3 = 'result'
word4 = 'reduced'
word5 = 'absorption'
embedding_vector1 = embeddings_index.get(word1)
embedding_vector2 = embeddings_index.get(word2)
embedding_vector3 = embeddings_index.get(word3)
embedding_vector4 = embeddings_index.get(word4)
embedding_vector5 = embeddings_index.get(word5)

# print embedding_vector1
# print embedding_vector2
# print embedding_vector3
# print embedding_vector4
# print embedding_vector5

word5 = 'use'
word6 = 'one'
word7 = 'may'
word8 = 'in'
word9 = 'the'
word10 = 'of'
word11 = 'with'

embedding_vector5 = embeddings_index.get(word5)
embedding_vector6 = embeddings_index.get(word6)
embedding_vector7 = embeddings_index.get(word7)
embedding_vector8 = embeddings_index.get(word8)
embedding_vector9 = embeddings_index.get(word9)
embedding_vector10 = embeddings_index.get(word10)
embedding_vector11 = embeddings_index.get(word11)

print word5, ' ', embedding_vector5
print word6, ' ', embedding_vector6
print word7, ' ', embedding_vector7
print word8, ' ', embedding_vector8
print word9, ' ', embedding_vector9
print word10, ' ', embedding_vector10
print word11, ' ', embedding_vector11