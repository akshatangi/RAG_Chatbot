import faiss
import numpy as np

# create dummy vectors
d = 64                 # dimension
xb = np.random.random((1000, d)).astype('float32')
xq = np.random.random((5, d)).astype('float32')

# build index
index = faiss.IndexFlatL2(d)   # L2 distance index
index.add(xb)                  # add vectors to index

# search
D, I = index.search(xq, 3)     # search top-3 nearest neighbors

print("Distances:\n", D)
print("Indices:\n", I)
