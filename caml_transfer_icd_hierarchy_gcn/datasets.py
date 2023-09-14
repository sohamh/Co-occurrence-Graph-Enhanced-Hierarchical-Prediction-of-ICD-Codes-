"""
    Data loading methods
"""
from collections import defaultdict
import csv
import math
import numpy as np
import sys
import pickle
from learn.arguments import args_parser
from icdcodex import icd2vec, hierarchy
import random

from constants import *

class Batch:
    """
        This class and the data_generator could probably be replaced with a PyTorch DataLoader
    """
    def __init__(self, desc_embed):
        self.docs = []
        self.labels = []
        self.hadm_ids = []
        self.code_set = set()
        self.length = 0
        self.max_length = MAX_LENGTH
        self.desc_embed = desc_embed
        self.descs = []

    def add_instance(self, row, ind2c, c2ind, w2ind, dv_dict, num_labels):
        """
            Makes an instance to add to this batch from given row data, with a bunch of lookups
        """
        labels = set()
        hadm_id = int(row[1])
        text = row[2]
        length = int(row[4])
        cur_code_set = set()
        labels_idx = np.zeros(num_labels)
        labelled = False
        desc_vecs = []
        #get codes as a multi-hot vector
        for l in row[3].split(';'):
            if l in c2ind.keys():
                code = int(c2ind[l])
                labels_idx[code] = 1
                cur_code_set.add(code)
                labelled = True
        if not labelled:
            return
        if self.desc_embed:
            for code in cur_code_set:
                l = ind2c[code]
                if l in dv_dict.keys():
                    #need to copy or description padding will get screwed up
                    desc_vecs.append(dv_dict[l][:])
                else:
                    desc_vecs.append([len(w2ind)+1])
        #OOV words are given a unique index at end of vocab lookup
        text = [int(w2ind[w]) if w in w2ind else len(w2ind)+1 for w in text.split()]
        #truncate long documents
        if len(text) > self.max_length:
            text = text[:self.max_length]

        #build instance
        self.docs.append(text)
        self.labels.append(labels_idx)
        self.hadm_ids.append(hadm_id)
        self.code_set = self.code_set.union(cur_code_set)
        if self.desc_embed:
            self.descs.append(pad_desc_vecs(desc_vecs))
        #reset length
        self.length = min(self.max_length, length)

    def pad_docs(self):
        #pad all docs to have self.length
        padded_docs = []
        for doc in self.docs:
            if len(doc) < self.length:
                doc.extend([0] * (self.length - len(doc)))
            padded_docs.append(doc)
        self.docs = padded_docs

    def to_ret(self):
        return np.array(self.docs), np.array(self.labels), np.array(self.hadm_ids), self.code_set,\
               np.array(self.descs)

def pad_desc_vecs(desc_vecs):
    #pad all description vectors in a batch to have the same length
    desc_len = max([len(dv) for dv in desc_vecs])
    pad_vecs = []
    for vec in desc_vecs:
        if len(vec) < desc_len:
            vec.extend([0] * (desc_len - len(vec)))
        pad_vecs.append(vec)
    return pad_vecs

def data_generator(filename, dicts, batch_size, num_labels, desc_embed=False, version='mimic3'):
    """
        Inputs:
            filename: holds data sorted by sequence length, for best batching
            dicts: holds all needed lookups
            batch_size: the batch size for train iterations
            num_labels: size of label output space
            desc_embed: true if using DR-CAML (lambda > 0)
            version: which (MIMIC) dataset
        Yields:
            np arrays with data for training loop.
    """
    ind2w, w2ind, ind2c, c2ind, dv_dict = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind'], dicts['dv']
    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        #header
        next(r)
        cur_inst = Batch(desc_embed)
        for row in r:
            #find the next `batch_size` instances
            if len(cur_inst.docs) == batch_size:
                cur_inst.pad_docs()
                yield cur_inst.to_ret()
                #clear
                cur_inst = Batch(desc_embed)
            cur_inst.add_instance(row, ind2c, c2ind, w2ind, dv_dict, num_labels)
        cur_inst.pad_docs()
        yield cur_inst.to_ret()

def load_vocab_dict(args, vocab_file):
    #reads vocab_file into two lookups (word:ind) and (ind:word)
    vocab = set()
    with open(vocab_file, 'r') as vocabfile:
        for i,line in enumerate(vocabfile):
            line = line.rstrip()
            if line != '':
                vocab.add(line.strip())
    #hack because the vocabs were created differently for these models
    if args.public_model and args.Y == 'full' and args.version == "mimic3" and args.model == 'conv_attn':
        ind2w = {i:w for i,w in enumerate(sorted(vocab))}
    else:
        ind2w = {i+1:w for i,w in enumerate(sorted(vocab))}
    w2ind = {w:i for i,w in ind2w.items()}
    return ind2w, w2ind

def diff(list1, list2):
    c = set(list1).union(set(list2))  # or c = set(list1) | set(list2)
    d = set(list1).intersection(set(list2))  # or d = set(list1) & set(list2)
    return list(c - d)



def load_lookups(args, desc_embed=False):
    """
        Inputs:
            args: Input arguments
            desc_embed: true if using DR-CAML
        Outputs:
            vocab lookups, ICD code lookups, description lookup, description one-hot vector lookup
    """
    #get vocab lookups
    ind2w, w2ind = load_vocab_dict(args, args.vocab)

    #get code and description lookups
    if args.Y == 'full':
        ind2c, desc_dict = load_full_codes(args.data_path, version=args.version)
    else:
        codes = set()
        with open("%s/TOP_%s_CODES.csv" % (MIMIC_3_DIR, str(args.Y)), 'r') as labelfile:
            lr = csv.reader(labelfile)
            for i,row in enumerate(lr):
                codes.add(row[0])
        ind2c = {i:c for i,c in enumerate(sorted(codes))}
        desc_dict = load_code_descriptions()
    c2ind = {c:i for i,c in ind2c.items()}

    #get description one-hot vector lookup
    if desc_embed:
        dv_dict = load_description_vectors(args.Y, version=args.version)
    else:
        dv_dict = None

    ##### added by Soha
    if(args.subcode==True):
        with open("/material/unk_key", "rb") as fp:  # Unpickling
            s = pickle.load(fp)
        corrected_icd9 = []
        ind2c2=ind2c.copy()
        c2ind2=c2ind.copy()
        for i in c2ind:
            corrected_icd9.append(i.replace('.', ''))
        count=0;
        for c in s:
            # ind = corrected_icd9.index(c)
            # ind=[i for i,elem in enumerate(corrected_icd9) if elem==c]
            ind = []
            for i,elem in enumerate(corrected_icd9):
                if elem==c:
                    ind.append(i)
            if(len(ind)==1):
                c2ind.pop(ind2c2.get(ind[0]))
                ind2c.pop(ind[0])
            else:
                count = count + 1
                print(c + " " + str(i))
                for i in ind:
                    try:
                        c2ind.pop(ind2c2.get(i))
                        ind2c.pop(i)
                    except:
                        print("***")

        print("number of repititive unknown codes: "+str(count))
        code2ind = dict(zip(c2ind.keys(), list(range(0, len(c2ind)))))
        ind2code = dict(zip(list(range(0, len(c2ind))), c2ind.keys()))
        dicts = {'ind2w': ind2w, 'w2ind': w2ind, 'ind2c': ind2code, 'c2ind': code2ind, 'desc': desc_dict, 'dv': dv_dict}
    else:
        dicts = {'ind2w': ind2w, 'w2ind': w2ind, 'ind2c': ind2c, 'c2ind': c2ind, 'desc': desc_dict, 'dv': dv_dict}
    if(args.ontology):
        embedder = icd2vec.Icd2Vec(num_embedding_dimensions=50, workers=-1)
        embedder.fit(*hierarchy.icd9())
        d = list(code2ind.keys())
        codes = [s.replace('.', '') for s in d]
        codes_of_interest = codes
        embedding = embedder.to_vec(codes_of_interest)
        data = {'embedding' : embedding, 'codes' : codes_of_interest}
        f = open("/material/embedding.pkl","wb")
        pickle.dump(data,f)
        f.close()

    if (args.transfer):
        random.seed(53)
        rndint_9 = random.sample(range(0, len(dicts['c2ind'])), math.floor(len(dicts['c2ind']) / 2))
        all = list(range(0, math.floor(len(dicts['c2ind']))))
        rndint_10 = diff(all, rndint_9)
        if (args.transfer_10):
            keys10 = [k for k, v in c2ind.items() if rndint_10.count(v) > 0]
            code2ind = dict(zip(keys10, list(range(0, len(rndint_10)))))
            ind2code = dict(zip(list(range(0, len(rndint_10))), keys10))
            # ont_emb=self.ont_emb(code2ind.keys(), version=10)
            dicts = {'ind2w': ind2w, 'w2ind': w2ind, 'ind2c': ind2code, 'c2ind': code2ind, 'desc': desc_dict,
                     'dv': dv_dict}
        else:
            keys9 = [k for k, v in c2ind.items() if rndint_9.count(v) > 0]
            code2ind = dict(zip(keys9, list(range(0, len(rndint_9)))))
            ind2code = dict(zip(list(range(0, len(rndint_9))), keys9))
            dicts = {'ind2w': ind2w, 'w2ind': w2ind, 'ind2c': ind2code, 'c2ind': code2ind, 'desc': desc_dict,
                     'dv': dv_dict}
    # For tranfer learning, begining with ICD 9
    return dicts

def load_full_codes(train_path, version='mimic3'):
    """
        Inputs:
            train_path: path to train dataset
            version: which (MIMIC) dataset
        Outputs:
            code lookup, description lookup
    """
    #get description lookup
    desc_dict = load_code_descriptions(version=version)
    #build code lookups from appropriate datasets
    if version == 'mimic2':
        ind2c = defaultdict(str)
        codes = set()
        with open('%s/proc_dsums.csv' % MIMIC_2_DIR, 'r') as f:
            r = csv.reader(f)
            #header
            next(r)
            for row in r:
                codes.update(set(row[-1].split(';')))
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i:c for i,c in enumerate(sorted(codes))})
    else:
        codes = set()
        for split in ['train', 'dev', 'test']:
            with open(train_path.replace('train', split), 'r') as f:
                lr = csv.reader(f)
                next(lr)
                for row in lr:
                    for code in row[3].split(';'):
                        codes.add(code)
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i:c for i,c in enumerate(sorted(codes))})
    return ind2c, desc_dict

def reformat(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits, 
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code

def load_code_descriptions(version='mimic3'):
    #load description lookup from the appropriate data files
    desc_dict = defaultdict(str)
    if version == 'mimic2':
        with open('%s/MIMIC_ICD9_mapping' % MIMIC_2_DIR, 'r') as f:
            r = csv.reader(f)
            #header
            next(r)
            for row in r:
                desc_dict[str(row[1])] = str(row[2])
    else:
        with open("%s/D_ICD_DIAGNOSES.csv" % (DATA_DIR), 'r') as descfile:
            r = csv.reader(descfile)
            #header
            next(r)
            for row in r:
                code = row[1]
                desc = row[-1]
                desc_dict[reformat(code, True)] = desc
        with open("%s/D_ICD_PROCEDURES.csv" % (DATA_DIR), 'r') as descfile:
            r = csv.reader(descfile)
            #header
            next(r)
            for row in r:
                code = row[1]
                desc = row[-1]
                if code not in desc_dict.keys():
                    desc_dict[reformat(code, False)] = desc
        with open('%s/ICD9_descriptions' % DATA_DIR, 'r') as labelfile:
            for i,row in enumerate(labelfile):
                row = row.rstrip().split()
                code = row[0]
                if code not in desc_dict.keys():
                    desc_dict[code] = ' '.join(row[1:])
    return desc_dict

def load_description_vectors(Y, version='mimic3'):
    #load description one-hot vectors from file
    dv_dict = {}
    if version == 'mimic2':
        data_dir = MIMIC_2_DIR
    else:
        data_dir = MIMIC_3_DIR
    with open("%s/description_vectors.vocab" % (data_dir), 'r') as vfile:
        r = csv.reader(vfile, delimiter=" ")
        #header
        next(r)
        for row in r:
            code = row[0]
            vec = [int(x) for x in row[1:]]
            dv_dict[code] = vec
    return dv_dict
