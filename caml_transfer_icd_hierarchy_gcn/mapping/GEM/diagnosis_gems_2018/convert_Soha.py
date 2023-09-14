import pickle
##### The code to save the dictionary and label from GEM text files
# dict_path="/mnt/data2/soha/codes/caml-mimic-master_Soha_01/mapping/GEM/diagnosis_gems_2018/2018_I9gem.txt"
# with open(dict_path) as f:
#     dict={}
#     label=[]
#     lines = f.readlines()
#     for l in lines:
#         l1=l.split(" ")
#         while "" in l1:
#             l1.remove("")
#         dict[l1[0]]=l1[1]
#         label.append(l1[2].strip('\n'))
# with open('/mnt/data2/soha/codes/caml-mimic-master_Soha_01/mapping/GEM/diagnosis_gems_2018/SOHA-GEMS9to10.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#         pickle.dump([dict, label], f)


#### now we use the saved dictionary to convert ICD 9 to ICD 10
with open('/mnt/data2/soha/codes/caml-mimic-master_Soha_01/mapping/GEM/diagnosis_gems_2018/SOHA-GEMS9to10.pkl',"rb") as f:  # Python 3: open(..., 'rb')
    dict, label = pickle.load(f)

ICD9="/mnt/data2/soha/codes/caml-mimic-master_Soha_01/mapping/in.txt"#['0300']
ICD10="/mnt/data2/soha/codes/caml-mimic-master_Soha_01/mapping/out"

f3=open(ICD9, 'r')
print ("NOTICE:Converting ICD9 codes to ICD10 codes!")
count=0
with open(ICD10 + "_ICD10.txt",'w') as output2:
    for key in f3.readlines():
        key=key.replace("\n","")
        key=key.replace(".", "")
        val=dict.get(key)
        output2.write('%s\n' % val)
        if(val is None):
            count=count+1
print("How many codes not found? "+str(count))