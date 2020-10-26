import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
import statistics
from collections import Counter
import csv

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


def plot_loss(plt_name,losses,save_path='./'):
    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(save_path+plt_name, bbox_inches='tight')
    plt.clf()

def get_emb_from_txt(source_path,verbose=False):
    num_lines = cntline_txtfile(source_path)
    with open(source_path, encoding="utf-8", mode="r") as textFile:
        for i, line in enumerate(textFile):
            if i == 0:  
                line = line.split() 
                emb_dim = len(line[1:])
            elif i == 1:
                line = line.split() 
                if len(line[1:]) == emb_dim: 
                    startline = 0 
                    num_vocab = num_lines
                    break
                else: 
                    emb_dim = len(line[1:])
                    startline = 1 
                    num_vocab = num_lines - 1 
                    break
    if verbose: 
        print('num_vocab',num_vocab)
        print('startline',startline)
    
    emb = np.zeros((num_vocab,emb_dim))
    if verbose: print('emb.shape',emb.shape)
    w2id = {}
    i = 0 
    with open(source_path, encoding="utf-8", mode="r") as textFile:
        for j, line in enumerate(textFile):
            if verbose: print('i',i,'j',j)
            if j >= startline:
                line = line.split()
                if verbose: print('cond check: len(line[1:]) == emb_dim',len(line[1:]) == emb_dim)
                if len(line[1:]) == emb_dim:
                    if verbose: print('cond check:line[0] in w2id.keys()',line[0] in w2id.keys())
                    if line[0] in w2id.keys():
                        if verbose: print('dup word:',line[0])
                    else:
                        w2id[line[0]] = i 
                        emb[i] = np.array(line[1:], dtype=np.float32)
                        i+=1
                else:
                    if verbose: print('i where len(line[1:]) != dim:',i)
                    if verbose: print('word where len(line[1:]) != dim:',line[0])
                    emb = np.delete(emb,i,0)
    return emb, w2id


def write_row(row, name, path="./", round=True):
    if round:
        row = [np.round(i, 2) for i in row]
    f = path + name + ".csv"
    with open(f, "a+") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=",")
        filewriter.writerow(row)

def cntline_txtfile(filename):
    lines = 0
    for line in open(filename):
        lines += 1
    return lines

def load_emb_w2id(embpathlist, verbose=False):
    num_emb = len(embpathlist)
    emblist = []
    w2idlist = []
    for i in range(num_emb):
        emb_tosave, w2id_tosave = get_emb_from_txt(embpathlist[i],verbose)
        emblist.append(emb_tosave)
        w2idlist.append(w2id_tosave)
    return emblist, w2idlist
