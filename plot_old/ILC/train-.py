#!/usr/bin/env python

import numpy as np
import random
import pickle
import argparse
#import cupy
import chainer
from chainer.backends import cuda
from chainer import Function, report, training, utils, Variable
from chainer import iterators, optimizers, serializers, dataset
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import matplotlib
matplotlib.use('Agg')


def load_dataset(fnames, sanity_check=False):
    dat = []
    lab = []
    num = []
    cnt = -1
    for c, fname in enumerate(fnames):
        k = 0
        with open(fname) as fin:
            while True:
                line = fin.readline()
                if not line:
                    break
                if line.rstrip() == '-------------------':
                    k += 1
                    if sanity_check and k == 101:
                        break
                    cnt += 1
                    line = fin.readline()
                    axis = np.array([float(v) for v in line.rstrip().split()[2:]], dtype=np.float32)
                    dat.append({'axis': axis, 'part': []})
                    lab.append(c)
                    num.append(0)
                    continue
                part = np.array([float(v) for v in line.strip().split()], dtype=np.float32)
                dat[cnt]['part'].append(part)

    max_part = 0
    for d in dat:
        if max_part < len(d['part']):
            max_part = len(d['part'])

    for i in range(len(dat)):
        a = np.zeros((max_part, 7), dtype=np.float32)
        l = len(dat[i]['part'])
        if l == 0:
            l = 1
        else:
            a[:l,:] = np.array(dat[i]['part'])
        dat[i]['part'] = a
        num[i] = l
    
    return dat, num, lab



class RCNPDataset(dataset.DatasetMixin):
    def __init__(self, dat, num, lab, split='train'):
        if split == 'train':
            self.dat = [dat[i] for i in ind1]
            self.num = np.array([num[i] for i in ind1], dtype=np.float32)
            self.lab = [lab[i] for i in ind1]
        elif split == 'validation':
            self.dat = [dat[i] for i in ind2]
            self.num = np.array([num[i] for i in ind2], dtype=np.float32)
            self.lab = [lab[i] for i in ind2]
        elif split == 'test':
            self.dat = [dat[i] for i in ind3]
            self.num = np.array([num[i] for i in ind3], dtype=np.float32)
            self.lab = [lab[i] for i in ind3]
        else:
            raise ValueError('split must be either train, validation, or test')
        
        lab = np.array(self.lab, dtype=np.int32)
        print('%s: %d (%d %d %d)' % (split, len(lab), np.sum(lab==0), np.sum(lab==1), np.sum(lab==2)))
        
    def __len__(self):
        return len(self.dat)
    
    def get_example(self, i):
        return self.dat[i]['axis'], self.dat[i]['part'], self.num[i], self.lab[i]            

     
    
class BaselineModel(Chain):
    def __init__(self, n_units=100, n_out=3):
        super(BaselineModel, self).__init__()
        with self.init_scope():
            self.lp1 = L.Linear(None, n_units)
            self.lp2 = L.Linear(None, n_units)
            self.lc1 = L.Linear(None, n_units)
            self.lc2 = L.Linear(None, n_units)
            self.lc3 = L.Linear(None, n_units)
            self.lc4 = L.Linear(None, n_units)
            self.lc5 = L.Linear(None, n_out)
            
    def __call__(self, a, p, n):
        n_bat, n_par, n_dim = p.shape 
        nonl = F.tanh
        h = F.reshape(p, (-1, n_dim))
        h = nonl(self.lp1(h))
        h = nonl(self.lp2(h))
        h = F.reshape(h, (n_bat, n_par, -1))
        h = F.sum(h, axis=1) / n.reshape((-1, 1))
        h = F.concat((h, a), axis=1)
        h = nonl(self.lc1(h))
        h = nonl(self.lc2(h))
        h = nonl(self.lc3(h))
        h = nonl(self.lc4(h))
        return self.lc5(h)



def evaluate(fname, ds):
    import sklearn.metrics
    
    model = BaselineModel(n_units=30, n_out=3)
    chainer.serializers.load_npz(fname, model)
    model.to_cpu()
    
    bsize = 2048
    it = iterators.SerialIterator(ds, bsize, repeat=False, shuffle=False)
    y = np.ndarray((len(ds),3))
    t = np.ndarray((len(ds,)), dtype=np.int32)
    try:
        i = 0
        while True:
            bat = it.next()
            bat = dataset.convert.concat_examples(bat, device=-1)
            res = F.softmax(model(bat[0], bat[1], bat[2])).data
            l   = res.shape[0]
            y[bsize*i:bsize*i+l,:] = F.softmax(model(bat[0], bat[1], bat[2])).data
            t[bsize*i:bsize*i+l  ] = bat[3]
            i += 1
    except StopIteration as e:
        print('iteration stopped')
    
    z = np.argmax(y, axis=1)
    print(z[:100])
    print(t[:100])
    print('ACC:', sklearn.metrics.accuracy_score(t, z))
    print('Confusion Matrix:\n', sklearn.metrics.confusion_matrix(t, z))
    with open('out.txt', 'w') as fout:
        for i in range(len(y)):
            fout.write('%d %f %f %f\n' % (t[i], y[i,0], y[i,1], y[i,2]))


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rebuild_dataset', action='store_true')
    parser.add_argument('-d', '--device'  , type=int, default=-1)
    parser.add_argument('-e', '--evaluate', type=str, default=None)
    args = parser.parse_args()
    
    if args.rebuild_dataset:
        fnames = ['../data/ILC/bb_data1.txt', '../data/ILC/cc_data1.txt', '../data/ILC/uds_data1.txt']
        dat, num, lab = load_dataset(fnames, sanity_check=False)

        l = len(dat)
        ind = list(range(l))
        random.shuffle(ind)
        s1 = int(l * 0.6)
        s2 = int(l * 0.8)
        ind1 = ind[:s1]
        ind2 = ind[s1:s2]
        ind3 = ind[s2:]
        #print(ind1,ind2,ind3)
        
        ds_train = RCNPDataset(dat, num, lab, split='train')
        ds_valid = RCNPDataset(dat, num, lab, split='validation')
        ds_test  = RCNPDataset(dat, num, lab, split='test')
    
        with open('../data/ILC/rcnpdataset_train.pickle', 'wb') as fout:
            pickle.dump(ds_train, fout)
        with open('../data/ILC/rcnpdataset_valid.pickle', 'wb') as fout:
            pickle.dump(ds_valid, fout)
        with open('../data/ILC/rcnpdataset_test.pickle', 'wb') as fout:
            pickle.dump(ds_test, fout)
    else:
        with open('../data/ILC/rcnpdataset_train.pickle', 'rb') as fin:
            ds_train = pickle.load(fin)
        with open('../data/ILC/rcnpdataset_valid.pickle', 'rb') as fin:
            ds_valid = pickle.load(fin)
        with open('../data/ILC/rcnpdataset_test.pickle', 'rb') as fin:
            ds_test = pickle.load(fin)
        
        lab = np.array(ds_train.lab, dtype=np.int32)
        print('train: %d (%d %d %d)' % (len(lab), np.sum(lab==0), np.sum(lab==1), np.sum(lab==2)))
        lab = np.array(ds_valid.lab, dtype=np.int32)
        print('valid: %d (%d %d %d)' % (len(lab), np.sum(lab==0), np.sum(lab==1), np.sum(lab==2)))
        lab = np.array(ds_test.lab, dtype=np.int32)
        print('test: %d (%d %d %d)' % (len(lab), np.sum(lab==0), np.sum(lab==1), np.sum(lab==2)))
   
    if args.evaluate is not None:
        evaluate(args.evaluate, ds_test)
        exit(0)

    batchsize = 1024
    it_train = iterators.SerialIterator(ds_train, batchsize)
    it_valid = iterators.SerialIterator(ds_valid, batchsize, False, False)
    it_test  = iterators.SerialIterator(ds_test , batchsize, False, False)

    model = BaselineModel(n_units=30, n_out=3)
    if args.device >= 0:
        model.to_gpu(args.device)
        cuda.get_device_from_id(args.device).use()
    
    model = L.Classifier(model)
    optimizer = optimizers.Adam(alpha=0.01)
    optimizer.setup(model)
    updater = training.updaters.StandardUpdater(it_train, optimizer, device=args.device)

    trainer = training.Trainer(updater, (1000, 'epoch'), out='result')
    
    trainer.extend(extensions.ExponentialShift('alpha', 0.5), trigger=(200, 'epoch'))
    
    trainer.extend(extensions.Lscheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)ogReport())
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'), trigger=(20, 'epoch'))
    trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}'), trigger=(20, 'epoch'))
    trainer.extend(extensions.Evaluator(it_valid, model, device=args.device), trigger=(1, 'epoch'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']
        ), trigger=(200, 'iteration'))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))
    
    trainer.run()
    
