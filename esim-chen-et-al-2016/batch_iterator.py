import cPickle as pkl
import gzip
import numpy as np
import random
import math

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target, label,
                 dict,
                 batch_size=128,
                 n_words=-1,
                 shuffle=True):
        self.source = fopen(source, 'r')
        self.target = fopen(target, 'r')
        self.label = fopen(label, 'r')
        with open(dict, 'rb') as f:
            self.dict = pkl.load(f)
        self.batch_size = batch_size
        self.n_words = n_words
        self.shuffle = shuffle
        self.end_of_data = False

        self.source_buffer = []
        self.target_buffer = []
        self.label_buffer = []
        self.k = int(batch_size * 20 /3 )
        self.source_0=[]
        self.target_0=[]
        self.label_0=[]
        self.source_1 = []
        self.target_1 = []
        self.label_1 = []
        self.source_2 = []
        self.target_2 = []
        self.label_2 = []

        self.current=-1
        self.do_all = False
    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)
        self.label.seek(0)
    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        label = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'
        assert len(self.source_buffer) == len(self.label_buffer), 'Buffer size mismatch!'
        if len(self.source_buffer) == 0:
            proper_batch=False

            while True:
                try:
                    ss = self.source.readline()
                    if ss == "":
                        break
                    tt = self.target.readline()
                    if tt == "":
                        break
                    ll = self.label.readline()
                    if ll == "":
                        break
                    s = ss.strip().split()
                    t = tt.strip().split()
                    l = ll.strip()
                    if int(l) == 0:
                        self.source_0.append(s)
                        self.target_0.append(t)
                        self.label_0.append(l)
                        if len(self.label_0)>=self.k:
                            self.current=0
                            break
                    elif int(l) == 1:
                        self.source_1.append(s)
                        self.target_1.append(t)
                        self.label_1.append(l)
                        if len(self.label_1)>=self.k:
                            self.current=1
                            break
                    else:
                        self.source_2.append(s)
                        self.target_2.append(t)
                        self.label_2.append(l)
                        if len(self.label_2)>=self.k:
                            self.current=2
                            break

                except IOError:
                    self.end_of_data = True
                if self.current==-1:

                    self.do_all=True
                    if(len(self.label_0)>0):
                        self.current=0
                    elif (len(self.label_1)>0):
                        self.current=1
                    elif (len(self.label_2) > 0):
                        self.current=2
                    else:
                        self.do_all=False
                        self.end_of_data = False
                        self.reset()
                        raise StopIteration
            for k_ in xrange(self.k):
                try:


                    if self.current == 0 :
                        s=self.source_0.pop(0)
                        t=self.target_0.pop(0)
                        l=self.label_0.pop(0)
                    elif self.current == 1:
                        s = self.source_1.pop(0)
                        t = self.target_1.pop(0)
                        l = self.label_1.pop(0)
                    else:
                        s = self.source_2.pop(0)
                        t = self.target_2.pop(0)
                        l = self.label_2.pop(0)
                    self.source_buffer.append(s)
                    self.target_buffer.append(t)
                    self.label_buffer.append(l)
                except IndexError:
                    break
            if self.shuffle:
                # sort by target buffer
                tlen = np.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()
                # shuffle mini-batch
                tindex = []
                small_index = range(int(math.ceil(len(tidx)*1./self.batch_size)))
                random.shuffle(small_index)
                for i in small_index:
                    if (i+1)*self.batch_size > len(tidx):
                        tindex.extend(tidx[i*self.batch_size:])
                    else:
                        tindex.extend(tidx[i*self.batch_size:(i+1)*self.batch_size])

                tidx = tindex

                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]
                _lbuf = [self.label_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.target_buffer = _tbuf
                self.label_buffer = _lbuf
        if len(self.source_buffer) == 0 or len(self.target_buffer) == 0 or len(self.label_buffer) == 0:
                self.end_of_data = False
                self.reset()
                raise StopIteration


        # actual work here
        while True:

            # read from source file and map to word index
            try:
                ss = self.source_buffer.pop(0)
            except IndexError:
                break

            ss.insert(0, '_BOS_')
            ss.append('_EOS_')
            ss = [self.dict[w] if w in self.dict else 1
                  for w in ss]
            if self.n_words > 0:
                ss = [w if w < self.n_words else 1 for w in ss]

            # read from source file and map to word index
            tt = self.target_buffer.pop(0)
            tt.insert(0, '_BOS_')
            tt.append('_EOS_')
            tt = [self.dict[w] if w in self.dict else 1
                  for w in tt]
            if self.n_words > 0:
                tt = [w if w < self.n_words else 1 for w in tt]

            # read label
            ll = self.label_buffer.pop(0)

            source.append(ss)
            target.append(tt)
            label.append(ll)

            if len(source) >= self.batch_size or \
                    len(target) >= self.batch_size or \
                    len(label) >= self.batch_size:
                break

        if len(source) <= 0 or len(target) <= 0 or len(label) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        return source, target, label
