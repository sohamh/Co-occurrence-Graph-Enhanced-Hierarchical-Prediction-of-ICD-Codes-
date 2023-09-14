import datasets
from learn.arguments import args_parser
from icdcodex import icd2vec, hierarchy
import pickle

args = args_parser()
desc_embed = args.lmbda > 0
dicts = datasets.load_lookups(args, desc_embed=desc_embed)
d=list(dicts['c2ind'].keys())
icd9=[s.replace('.', '') for s in d]
#### now we use the saved dictionary to convert ICD 9 to ICD 10
with open('/mnt/data2/soha/codes/caml-mimic-master_Soha_01/mapping/GEM/diagnosis_gems_2018/SOHA-GEMS9to10.pkl',"rb") as f:  # Python 3: open(..., 'rb')
    dict, label = pickle.load(f)
ICD10="/mnt/data2/soha/codes/caml-mimic-master_Soha_01/mapping/out"
count=0
icd10=[]
with open(ICD10 + "_ICD10.txt",'w') as output2:
    for key in icd9:
        val=dict.get(key)
        output2.write('%s\n' % val)
        if(val is None):
            count=count+1
        else:
            icd10.append(val)

print("How many codes not found? "+str(count))

### Now lets embedd the ICD 10, and count how many will be lost?
icd_10_cm_hierarchy, icd_10_cm_codes = hierarchy.icd10cm("2020")
embedder = icd2vec.Icd2Vec(num_embedding_dimensions=50, workers=-1)
embedder.fit(*hierarchy.icd10cm("2020"))
codes_of_interest = icd10#["0330", "0340", "9101"]


### to overcome the problem with icd 10 with dot codes and also count how many mis-matches we will have
corrected_icd10 = []
for i in icd_10_cm_codes:
    corrected_icd10.append(i.replace('.', ''))
mimic_icd10=[]
count2=0
for c in icd10:
    b=True
    try:
        ind=corrected_icd10.index(c)
    except ValueError:
        count2=count2+1
        b=False
    if(b):
        mimic_icd10.append(icd_10_cm_codes[ind])
    else:
        mimic_icd10.append(icd_10_cm_codes[ind]) #lets replace non existing ones with the previous one?
### To find the embeddings of teh existing codes
embedding = embedder.to_vec(mimic_icd10)
print('hello')
### to store the embeddings:
data = {'embedding' : embedding, 'codes' : mimic_icd10}
f = open("/mnt/data2/soha/codes/caml-mimic-master_Soha_01/embedding_ind10_50d.pkl","wb")
pickle.dump(data,f)
f.close()