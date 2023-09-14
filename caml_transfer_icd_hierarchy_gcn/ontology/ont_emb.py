import datasets
from learn.arguments import args_parser
from icdcodex import icd2vec, hierarchy
import pickle

args = args_parser()
desc_embed = args.lmbda > 0
dicts = datasets.load_lookups(args, desc_embed=desc_embed)
d=list(dicts['c2ind'].keys())
codes=[s.replace('.', '') for s in d]

with open("/mnt/data2/soha/codes/caml-mimic-master_Soha_01/unk_key", "rb") as fp:   # Unpickling
    s = pickle.load(fp)
[codes.remove(ss) for ss in s]

# workers=-1 parallelizes the node2vec algorithm across all available CPUs
embedder = icd2vec.Icd2Vec(num_embedding_dimensions=50, workers=-1)
embedder.fit(*hierarchy.icd10cm(2018))
codes_of_interest = codes#["0330", "0340", "9101"]
embedding = embedder.to_vec(codes_of_interest)

### to load the embeddings
# with open("/mnt/data2/soha/codes/caml-mimic-master_Soha_01/embedding.pkl", "rb") as fp:   # Unpickling
#     data = pickle.load(fp)
#     emb=data['embedding']
#     codes=data['codes']

### to store the embeddings:
# data = {'embedding' : embedding, 'codes' : codes_of_interest}
# f = open("/mnt/data2/soha/codes/caml-mimic-master_Soha_01/embedding_10.pkl","wb")
# pickle.dump(data,f)
# f.close()




