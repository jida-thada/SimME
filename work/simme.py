###### ------------ Param --------------########
import argparse
from nltk.corpus import wordnet as wn
from simme_utils import *
import sys
import os
import timeit
import pandas as pd
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import itertools  
import random
from torch.utils import data as torch_data
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def save_pp_embw2id(target_path,emb,w2id):
    # ---- save emb and w2id
    emb = emb.data.numpy()
    target_file = open(target_path, "w")
    for i in range(emb.shape[0]):
        l = list(w2id.keys())[list(w2id.values()).index(i)]+' ' + ' '.join(map(str, emb[i]))
        target_file.write(l+'\n')

def cosdist_pairs(fn,header,w1colid,w2colid,embname,source_emblist,source_w2idlist,source_embdim):
    filepath = '../data/'+fn
    w1=[]; w2=[]
    with open(filepath, encoding="utf-8", mode="r") as textFile:
        if header=='y': next(textFile)
        for line in textFile:
            line = line.split()
            w1_tmp = line[w1colid]
            w2_tmp = line[w2colid]
            w1.append(w1_tmp)
            w2.append(w2_tmp)
    pairs = list(map(lambda x, y:(x,y), w1, w2))

    # ---- adjust for 'unk' 
    emb_s = source_emblist
    w2id_s = source_w2idlist
    for i in range(len(emb_s)):
        emb_dim = int(source_embdim[i])
        if 'unk' not in w2id_s[i]:
            w2id_s[i]['unk'] = len(w2id_s[i])
            emb_s[i] = np.concatenate((emb_s[i],np.array([np.zeros(emb_dim)])), axis=0)

    # Gen data for pp training
    example_id = 0
    cos_dist = nn.CosineSimilarity(dim=1, eps=1e-6)
    embnn = []
    for si in range(len(emb_s)):
        emb = emb_s[si]; w2id = w2id_s[si]
        emb_dim = int(source_embdim[si])
        emb_si = nn.Embedding(emb.shape[0], emb_dim)
        emb_si.weight = torch.nn.Parameter(torch.FloatTensor(emb))
        embnn.append(emb_si)

    # ---- gen for each example
    for p in pairs:
        dist_s_cos = []; tkt_s_cos = [];
        for si in range(len(emb_s)):
            emb = embnn[si]; w2id = w2id_s[si];
            wid1 = w2id[p[0]] if p[0] in w2id.keys() else w2id['unk']
            wid2 = w2id[p[1]] if p[1] in w2id.keys() else w2id['unk']
            pid = [wid1,wid2]
            if p[0] in w2id.keys() and p[0] != 'unk' and p[1] in w2id.keys() and p[1] != 'unk': tkt = 0
            else: tkt = 1
            pairs_emb = emb(torch.tensor(pid, dtype=torch.long))
            wi = pairs_emb[0].view(1,emb_dim)
            wj = pairs_emb[1].view(1,emb_dim)
            dist_cosine = cos_dist(wi,wj)
            dist_s_cos.append(dist_cosine.detach().numpy())
            tkt_s_cos.append(tkt)
        row = []
        for j in range(len(emb_s)):
            row.extend((dist_s_cos[j][0],tkt_s_cos[j]))
        write_row(row, fn[:fn.rfind('.txt')]+'_cosdist_'+embname, '../data/', round=False)
        example_id += 1



class emb_corr_tkt(nn.Module):

    def __init__(self,vocab_size,emb_dim,w2id):
        super(emb_corr_tkt, self).__init__()
        #torch.manual_seed(1)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.w2id = w2id
        self.emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.cos_dist = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.verbose = False

    def forward(self,pairs_wid):
        self.pairs_vector = self.emb(pairs_wid)
        self.wi = self.pairs_vector[:,0,:]
        self.wj = self.pairs_vector[:,1,:]
        self.dist = self.cos_dist(self.wi,self.wj)
        if self.verbose: print('dist', self.dist)
        return self.dist


class emb_corr_tkt_train_pp:

    def __init__(self,param_embcorr,vocab,w2id,pairs,y,model_path,resume=False): #list_IDs,x_path,y_path,
        self.path = param_embcorr['path']
        self.title_name = param_embcorr['title_name']
        self.ep_num = param_embcorr['ep_num']
        self.bs = param_embcorr['bs']
        self.lr = param_embcorr['lr']
        self.emb_dim = param_embcorr['emb_dim']
        self.name = param_embcorr['progress_file']
        self.save_every = param_embcorr['save_every']
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.w2id = w2id
        self.resume = resume
        self.model_path = model_path
        params = {'shuffle': True,'num_workers': 4}
        params['batch_size'] = self.bs
        self.use_gpu =  torch.cuda.is_available()
        self.train_x = pairs
        self.train_y = y
        self.embcorrtrain()

    def dataloader(self,train_x,train_y):
        train_data=TensorDataset(torch.FloatTensor(train_x), torch.FloatTensor(train_y))
        train_ld=DataLoader(train_data, batch_size=self.bs, shuffle=True, drop_last=True)
        return train_ld

    def embcorrtrain(self):
        train_loader = self.dataloader(self.train_x,self.train_y)
        # resume
        if self.resume:
            checkpoint = torch.load(self.model_path)
            self.model = emb_corr_tkt(self.vocab_size, self.emb_dim, self.w2id)
            self.model = self.model.load_state_dict(checkpoint['model_state_dict']).cuda() if self.use_gpu else self.model.load_state_dict(checkpoint['model_state_dict'])
            print('self.model.load_state_dict(checkpoint[model_state_dict])')
            print(self.model.load_state_dict(checkpoint['model_state_dict']))
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
            self.optimizer = self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.ep_start = checkpoint['epoch']
            self.losses = checkpoint['loss']
        else:
            self.model = emb_corr_tkt(self.vocab_size, self.emb_dim, self.w2id).cuda() if self.use_gpu else emb_corr_tkt(self.vocab_size, self.emb_dim, self.w2id)
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
            self.ep_start = 0
            self.losses = []

        for epoch in range(self.ep_start, self.ep_num):
            self.verbose = True if epoch%self.save_every == 0 or epoch == self.ep_num-1 else False
            if self.verbose: print('ep:',epoch, end=' ')
            self.epoch_loss = 0
            for x, y in train_loader:
                self.x = x.type(torch.LongTensor)
                self.y = y.type(torch.FloatTensor)                
                self.num_source = int(len(self.y[0])/2)
                self.pred_dist = self.model(self.x)
                self.optimizer.zero_grad()# zero the parameter gradients
                self.loss = 0
                for j in range(self.num_source):
                    self.tkt = self.y[:,2*j+1]
                    self.d = self.y[:,2*j]
                    self.dist = (torch.Tensor([1]*len(self.tkt))- torch.squeeze(self.tkt))*torch.squeeze(self.d)
                    self.loss += torch.mean(( (torch.Tensor([1]*len(self.tkt))- torch.squeeze(self.tkt))*self.pred_dist - self.dist)**2, dim=-1)
                self.loss.backward()
                self.optimizer.step()
                self.epoch_loss += float(self.loss.data)
            self.losses.append(self.epoch_loss)

            # track by ep
            row = ['ep',epoch,self.epoch_loss]; write_row(row, self.name, self.path, round=False)

            if epoch != 0 and self.verbose: # Save model and emb
                torch.save({'epoch': epoch,'model_state_dict': self.model.state_dict(),'optimizer_state_dict': self.optimizer.state_dict(),'loss': self.losses}, self.model_path)
                save_pp_embw2id('./temp_emb/'+self.title_name+'ep'+str(epoch),self.model.emb.weight,self.w2id)
        return self.losses, self.model.emb.weight







def train_SimME(fn, embname,runid,param_csv,header,w1colid,w2colid,param_embcor,emb_dim):
    filepath = fn
    y_path = filepath[:filepath.rfind('.txt')]+'_cosdist_'+embname+'.csv'
    savepath = '../simme_output/'
    embname_tosave = embname+runid
    model_path = './temp_emb/'+embname_tosave+'.pt'
    param_embcor['title_name'] = embname_tosave
    param_embcor['progress_file'] = 'progress_'+param_embcor['title_name']
    param_csv['name'] = param_embcor['progress_file']

    w1=[]; w2=[]
    with open(filepath, encoding="utf-8", mode="r") as textFile:
        if header=='y': next(textFile)
        for line in textFile:
            line = line.split()
            w1_tmp = line[w1colid]
            w2_tmp = line[w2colid]
            w1.append(w1_tmp)
            w2.append(w2_tmp)
    vocab_cor = list(set(w1+w2))
    row = ['vocab size',len(vocab_cor)]
    write_row(row, param_csv['name'], param_csv['path'], round=False)
    if 'nan' in vocab_cor:
        row = ['nan in vocab_cor']
        write_row(row, param_csv['name'], param_csv['path'], round=False)

    # ---- w2id
    w2id_cor = {}
    wid = 0
    for w in vocab_cor:
        w2id_cor[w] = wid; wid+=1
    row = ['gather_vocab_w2id']
    write_row(row, param_csv['name'], param_csv['path'], round=False)

    # ---- adjust for 'unk'
    if 'unk' not in vocab_cor:
        w2id_cor['unk'] = len(vocab_cor)
        vocab_cor.append('unk')

    # ---- prepare data
    pairs = []
    with open(filepath, encoding="utf-8", mode="r") as f:
        for line in f:
            l = line.split()
            wid1 = w2id_cor[l[0]]
            wid2 = w2id_cor[l[1]]
            pairs.append([wid1,wid2])
    pairs = np.array(pairs) 

    df = pd.read_csv(y_path, header=None)
    y = [] 
    for i in range(df.shape[0]):
        all_y = []
        for j in range(df.shape[1]):
            all_y.append(df[j][i]) 
        y.append(all_y)

    # ---- train model    
    start_time = timeit.default_timer()
    train_model = emb_corr_tkt_train_pp(param_embcor,vocab_cor,w2id_cor,pairs,y,model_path)
    embcor_losses = train_model.losses
    embcor = train_model.model.emb.weight
    row = ['create emb_pp',timeit.default_timer() - start_time]
    write_row(row, param_csv['name'], param_csv['path'], round=False)

    # ---- save emb and w2id
    start_time = timeit.default_timer()
    save_pp_embw2id(savepath+embname_tosave+'.txt',embcor,w2id_cor)
    row = ['save emb_pp w2id_pp',timeit.default_timer() - start_time]
    write_row(row, param_csv['name'], param_csv['path'], round=False)

    plot_loss(param_embcor['title_name']+'_loss',embcor_losses,save_path='./plot/')






def main():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-embname',         type=str,                   help='')
    add_arg('--gendata',        action='store_true',        help='')
    add_arg('-pairs_filename',  type=str,                   help='')
    add_arg('-header',          type=str,   default='n',    help='y/n')
    add_arg('-w1colid',         type=int,   default=0,       help='')
    add_arg('-w2colid',         type=int,   default=1,       help='')
    add_arg('-source_filename', '--item',   action='store', nargs='*',  type=str,   help='')
    add_arg('-source_embdim',   nargs="+",  type=int,       help='200 200')
    add_arg('--train',          action='store_true',        help='')
    add_arg('-embdim',          type=int,   default=200,    help='')
    add_arg('-runid',           type=str,   default='',     help='')
    add_arg('-ep',              type=int,   default=200,    help='')
    add_arg('-lr',              type=float, default=15,     help='')
    add_arg('-bs',              type=int,   default=512,    help='')

    args = parser.parse_args()
    embname = args.embname
    gendata = args.gendata
    train = args.train

    param_csv = {}
    param_csv['path']='./log/'

    
    if gendata:
        fn = args.pairs_filename
        header = args.header
        w1colid = args.w1colid
        w2colid = args.w2colid
        source_fn_list = args.source_filename
        source_embdim = args.source_embdim
        num_emb = len(source_fn_list)
        source_namelist = []
        for i in range(num_emb):
            source_namelist.append(source_fn_list[i][source_fn_list[i].rfind('/')+1:])
        source_emblist, source_w2idlist = load_emb_w2id(source_fn_list)
        cosdist_pairs(fn,header,w1colid,w2colid,embname,source_emblist,source_w2idlist,source_embdim)
        print('Done gendata')

    
    if train:
        fn = args.pairs_filename
        header = args.header
        w1colid = args.w1colid
        w2colid = args.w2colid
        runid = args.runid
        emb_dim = args.embdim
        param_embcor={}
        param_embcor['ep_num'] = args.ep 
        param_embcor['bs'] = args.bs
        param_embcor['lr'] = args.lr
        param_embcor['path'] = param_csv['path']
        param_embcor['emb_dim'] = args.embdim
        param_embcor['save_every'] = int(args.ep/2)
        train_SimME(fn, embname,runid,param_csv,header,w1colid,w2colid,param_embcor,emb_dim)
    pass

if __name__ == '__main__':
    main()
    pass
