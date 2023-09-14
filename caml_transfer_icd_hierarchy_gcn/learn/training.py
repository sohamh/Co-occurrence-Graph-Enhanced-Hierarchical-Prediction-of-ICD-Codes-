"""
    Main training code. Loads data, builds the model, trains, tests, evaluates, writes outputs, etc.
"""
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import sys
sys.path.append('')

import csv
import os 
import numpy as np
import operator
import random
import sys
import time
from tqdm import tqdm
from collections import defaultdict

from constants import *
import datasets
import evaluation
import interpret
import persistence
import learn.models as models
import learn.tools as tools
from learn.arguments import args_parser

os.environ["CUDA_VISIBLE_DEVICES"] =str(0)

def main(args):
    start = time.time()
    args, model, optimizer, params, dicts = init(args)
    epochs_trained = train_epochs(args, model, optimizer, params, dicts)
    print("TOTAL ELAPSED TIME FOR %s MODEL AND %d EPOCHS: %f" % (args.model, epochs_trained, time.time() - start))

def init(args):
    """
        Load data, build model, create optimizer, create vars to hold metrics, etc.
    """
    #need to handle really large text fields
    csv.field_size_limit(sys.maxsize)

    #load vocab and other lookups
    desc_embed = args.lmbda > 0
    print("loading lookups...")
    dicts = datasets.load_lookups(args, desc_embed=desc_embed)

    model = tools.pick_model(args, dicts)
    print(model)

    if not args.test_model:
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    else:
        optimizer = None

    params = tools.make_param_dict(args)
    
    return args, model, optimizer, params, dicts

def train_epochs(args, model, optimizer, params, dicts):
    """
        Main loop. does train and test
    """
    metrics_hist = defaultdict(lambda: [])
    metrics_hist_te = defaultdict(lambda: [])
    metrics_hist_tr = defaultdict(lambda: [])

    test_only = args.test_model is not None
    evaluate = args.test_model is not None
    #train for n_epochs unless criterion metric does not improve for [patience] epochs
    for epoch in range(args.n_epochs):
        #only test on train/test set on very last epoch
        if epoch == 0 and not args.test_model:
            model_dir = os.path.join(MODEL_DIR, '_'.join([args.model, time.strftime('%b_%d_%H:%M:%S', time.localtime())]))
            os.mkdir(model_dir)
        elif args.test_model:
            model_dir = os.path.dirname(os.path.abspath(args.test_model))
        metrics_all = one_epoch(model, optimizer, args.Y, epoch, args.n_epochs, args.batch_size, args.data_path,
                                                  args.version, test_only, dicts, model_dir, 
                                                  args.samples, args.gpu, args.quiet)
        for name in metrics_all[0].keys():
            metrics_hist[name].append(metrics_all[0][name])
        for name in metrics_all[1].keys():
            metrics_hist_te[name].append(metrics_all[1][name])
        for name in metrics_all[2].keys():
            metrics_hist_tr[name].append(metrics_all[2][name])
        metrics_hist_all = (metrics_hist, metrics_hist_te, metrics_hist_tr)

        #save metrics, model, params
        persistence.save_everything(args, metrics_hist_all, model, model_dir, params, args.criterion, evaluate)

        if test_only:
            #we're done
            break

        if args.criterion in metrics_hist.keys():
            if early_stop(metrics_hist, args.criterion, args.patience):
                #stop training, do tests on test and train sets, and then stop the script
                print("%s hasn't improved in %d epochs, early stopping..." % (args.criterion, args.patience))
                test_only = True
                args.test_model = '%s/model_best_%s.pth' % (model_dir, args.criterion)
                model = tools.pick_model(args, dicts)
    return epoch+1

def early_stop(metrics_hist, criterion, patience):
    if not np.all(np.isnan(metrics_hist[criterion])):
        if len(metrics_hist[criterion]) >= patience:
            if criterion == 'loss_dev': 
                return np.nanargmin(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
            else:
                return np.nanargmax(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
    else:
        #keep training if criterion results have all been nan so far
        return False
        
def one_epoch(model, optimizer, Y, epoch, n_epochs, batch_size, data_path, version, testing, dicts, model_dir, 
              samples, gpu, quiet):
    """
        Wrapper to do a training epoch and test on dev
    """
    if not testing:
        losses, unseen_code_inds = train(model, optimizer, Y, epoch, batch_size, data_path, gpu, version, dicts, quiet)
        loss = np.mean(losses)
        print("epoch loss: " + str(loss))
    else:
        loss = np.nan
        if model.lmbda > 0:
            #still need to get unseen code inds
            print("getting set of codes not in training set")
            c2ind = dicts['c2ind']
            unseen_code_inds = set(dicts['ind2c'].keys())
            num_labels = len(dicts['ind2c'])
            with open(data_path, 'r') as f:
                r = csv.reader(f)
                #header
                next(r)
                for row in r:
                    unseen_code_inds = unseen_code_inds.difference(set([c2ind[c] for c in row[3].split(';') if c != '']))
            print("num codes not in train set: %d" % len(unseen_code_inds))
        else:
            unseen_code_inds = set()

    fold = 'test' if version == 'mimic2' else 'dev'
    if epoch == n_epochs - 1:
        print("last epoch: testing on test and train sets")
        testing = True
        quiet = False

    #test on dev
    metrics = test(model, Y, epoch, data_path, fold, gpu, version, unseen_code_inds, dicts, samples, model_dir,
                   testing)
    if testing or epoch == n_epochs - 1:
        print("\nevaluating on test")
        metrics_te = test(model, Y, epoch, data_path, "test", gpu, version, unseen_code_inds, dicts, samples, 
                          model_dir, True)
    else:
        metrics_te = defaultdict(float)
        fpr_te = defaultdict(lambda: [])
        tpr_te = defaultdict(lambda: [])
    metrics_tr = {'loss': loss}
    metrics_all = (metrics, metrics_te, metrics_tr)
    return metrics_all


def train(model, optimizer, Y, epoch, batch_size, data_path, gpu, version, dicts, quiet):
    """
        Training loop.
        output: losses for each example for this iteration
    """
    print("EPOCH %d" % epoch)
    num_labels = len(dicts['ind2c'])

    losses = []
    #how often to print some info to stdout
    print_every = 25

    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
    unseen_code_inds = set(ind2c.keys())
    desc_embed = model.lmbda > 0

    model.train()
    gen = datasets.data_generator(data_path, dicts, batch_size, num_labels, version=version, desc_embed=desc_embed)
    for batch_idx, tup in tqdm(enumerate(gen)):
        data, target, _, code_set, descs = tup
        data, target = Variable(torch.LongTensor(data)), Variable(torch.FloatTensor(target))
        unseen_code_inds = unseen_code_inds.difference(code_set)
        if gpu:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()

        if desc_embed:
            desc_data = descs
        else:
            desc_data = None

        output, loss, _ = model(data, target, desc_data=desc_data)

        loss.backward()
        optimizer.step()

        losses.append(loss.data[0])

        if not quiet and batch_idx % print_every == 0:
            #print the average loss of the last 10 batches
            print("Train epoch: {} [batch #{}, batch_size {}, seq length {}]\tLoss: {:.6f}".format(
                epoch, batch_idx, data.size()[0], data.size()[1], np.mean(losses[-10:])))
    return losses, unseen_code_inds

def unseen_code_vecs(model, code_inds, dicts, gpu):
    """
        Use description module for codes not seen in training set.
    """
    code_vecs = tools.build_code_vecs(code_inds, dicts)
    code_inds, vecs = code_vecs
    #wrap it in an array so it's 3d
    desc_embeddings = model.embed_descriptions([vecs], gpu)[0]
    #replace relevant final_layer weights with desc embeddings 
    model.final.weight.data[code_inds, :] = desc_embeddings.data
    model.final.bias.data[code_inds] = 0

def test(model, Y, epoch, data_path, fold, gpu, version, code_inds, dicts, samples, model_dir, testing):
    """
        Testing loop.
        Returns metrics
    """
    filename = data_path.replace('train', fold)
    print('file for evaluation: %s' % filename)
    num_labels = len(dicts['ind2c'])

    #initialize stuff for saving attention samples
    if samples:
        tp_file = open('%s/tp_%s_examples_%d.txt' % (model_dir, fold, epoch), 'w')
        fp_file = open('%s/fp_%s_examples_%d.txt' % (model_dir, fold, epoch), 'w')
        window_size = model.conv.weight.data.size()[2]

    y, yhat, yhat_raw, hids, losses = [], [], [], [], []
    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']

    desc_embed = model.lmbda > 0
    if desc_embed and len(code_inds) > 0:
        unseen_code_vecs(model, code_inds, dicts, gpu)

    model.eval()
    gen = datasets.data_generator(filename, dicts, 1, num_labels, version=version, desc_embed=desc_embed)
    for batch_idx, tup in tqdm(enumerate(gen)):
        data, target, hadm_ids, _, descs = tup
        data, target = Variable(torch.LongTensor(data), volatile=True), Variable(torch.FloatTensor(target))
        if gpu:
            data = data.cuda()
            target = target.cuda()
        model.zero_grad()

        if desc_embed:
            desc_data = descs
        else:
            desc_data = None

        #get an attention sample for 2% of batches
        get_attn = samples and (np.random.rand() < 0.02 or (fold == 'test' and testing))
        output, loss, alpha = model(data, target, desc_data=desc_data, get_attention=get_attn)

        output = F.sigmoid(output)
        output = output.data.cpu().numpy()
        losses.append(loss.data[0])
        target_data = target.data.cpu().numpy()
        if get_attn and samples:
            interpret.save_samples(data, output, target_data, alpha, window_size, epoch, tp_file, fp_file, dicts=dicts)

        #save predictions, target, hadm ids
        yhat_raw.append(output)
        output = np.round(output)
        y.append(target_data)
        yhat.append(output)
        hids.extend(hadm_ids)

    #close files if needed
    if samples:
        tp_file.close()
        fp_file.close()

    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    yhat_raw = np.concatenate(yhat_raw, axis=0)

    #write the predictions
    preds_file = persistence.write_preds(yhat, model_dir, hids, fold, ind2c, yhat_raw)
    #get metrics
    k = 5 if num_labels == 50 else [8,15]
    metrics = evaluation.all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
    evaluation.print_metrics(metrics)
    metrics['loss_%s' % fold] = np.mean(losses)
    return metrics
    

if __name__ == "__main__":
    args = args_parser()
    print(args)
    main(args)

