
import argparse
import logging

import torch
from torch.distributions import constraints
import torch.nn as nn
import torch.nn.functional as F

import torchtext.data as data

import pyro
import pyro.distributions as ds
from pyro import poutine
from pyro.infer import TraceEnum_ELBO
import pyro.optim

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)
pyro.enable_validation(True)

if True:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def get_args():
    pass


enum_style = "parallel"
#enum_style = "sequential"

TEXT = data.Field()
class Yahoo(data.Dataset):
    def __init__(self, text_field):
         fields = [('text', text_field)]
         examples = []
         for line in open("val.txt"):
            text = line.split()[:20]
            examples.append(data.Example.fromlist([text], fields))
         super(Yahoo, self).__init__(examples, fields)

yahoo = Yahoo(TEXT)
C = 10
TEXT.build_vocab(yahoo, max_size=5000)
V = len(TEXT.vocab)
train_iter = next(iter(data.BucketIterator(
         dataset=yahoo, batch_size=5000,
     sort_key=lambda x: len(x.text))))
config = (V, C)
BSZ = 64
nhid = 16

class NaiveBayes:
    def __init__(self, V, C):
        self.V = V
        self.C = C

    def guide(self, data):
        return self.model(data, observe=False)

    def model(self, data, observe=True):
        V, C = self.V, self.C
        N, T = data.shape

        pz = pyro.param("pz", torch.rand(C), constraint=constraints.simplex)
        px_z = pyro.param("px_z", torch.rand(C, V), constraint=constraints.simplex)

        with pyro.iarange("n", N, subsample_size=BSZ) as ind:
            z = pyro.sample(
                "z",
                ds.Categorical(pz.expand(BSZ, C)),
                infer = {
                    "enumerate": enum_style,
                },
            )
            if observe:
                """
                with pyro.iarange("T", T):
                    x = pyro.sample("x", ds.Categorical(px_z[z]), obs = data[ind].t())
                """
                #"""
                for t in pyro.irange("T", T):
                    x = pyro.sample(
                        f"x_{t}",
                        ds.Categorical(px_z[z]),
                        obs = data[ind, t],
                    )
                #"""


class HMM:
    def __init__(self, V, C):
        self.V = V
        self.C = C
        self.batch_x = False
        self.transition = True

        if self.transition:
            self.wlut = nn.Embedding(V, nhid)
            self.clut = nn.Embedding(C, nhid)
            self.rnn = nn.LSTM(
                input_size = nhid,
                hidden_size = nhid,
                num_layers = 1,
            )
            self.vproj = nn.Linear(nhid, V)
        if self.transition and self.batch_x:
            self.lm = 1
        if self.transition and not self.batch_x:
            def lm(x, c, h):
                x = self.wlut(x)
                c = self.clut(c)
                y, h = self.rnn((x + c).unsqueeze(1), h)
                import pdb; pdb.set_trace()
                return F.softmax(self.vproj(y)), h
            self.lm = lm

    def guide(self, data):
        return self.model(data, observe=False)

    def model(self, data, observe=True):
        V, C = self.V, self.C
        N, T = data.shape

        pz = pyro.param("pz", torch.rand(C), constraint=constraints.simplex)
        pz_z = pyro.param("px_x", torch.rand(C, C), constraint=constraints.simplex)
        if not self.transition:
            px_z = pyro.param("px_z", torch.rand(C, V), constraint=constraints.simplex)

        x_iarange = pyro.iarange("T", T)
        with pyro.iarange("n", N, subsample_size=BSZ) as ind:
            zs = []
            ht = None
            # Assume first token is bos (probably not true, lol)
            for t in range(1, T):
                dzt = ds.Categorical(pz_z[zt] if t > 1 else pz.expand(BSZ, C))
                zt = pyro.sample(
                    f"z_{t}", dzt,
                    infer = {"enumerate": enum_style}
                )
                zs.append(zt)
                if observe and not self.batch_x:
                    if self.transition:
                        # HACKY
                        z = 0*data[ind, t-1] + zt.squeeze(1)
                        x = data[ind, t-1] + 0*z
                        xt, ht = self.lm(x.view(-1), z.view(-1), ht)
                        dxt = ds.Categorical(xt)
                    else:
                        dxt = ds.Categorical(px_z[zt])
                    xt = pyro.sample(f"x_{t}", dxt, obs = data[ind, t])
            if observe and self.batch_x:
                z = torch.stack(zs) # T x N
                with pyro.iarange("T", T) as t:
                    dxt = ds.Categorical(px_z[z])
                    xt = pyro.sample(f"x", dxt, obs = data[ind].t())

class HSMM:
    def __init__(self, V, C):
        self.V = V
        self.C = C
        self.word_lut = nn.Embedding()
        self.class_lut = nn.Embedding()
        self.emission = nn.LSTM()

    def guide(self, data):
        return self.model(data, observe=False)

    def model(self, data, observe=True):
        V, C = self.V, self.C
        N, T = data.shape

        pz = pyro.param("pz", torch.rand(C), constraint=constraints.simplex)
        pz_z = pyro.param("px_x", torch.rand(C, C), constraint=constraints.simplex)
        px_z = pyro.param("px_z", torch.rand(C, V), constraint=constraints.simplex)

        x_iarange = pyro.iarange("T", T)
        with pyro.iarange("n", N, subsample_size=BSZ) as ind:
            zs = []
            for t in range(T):
                dzt = ds.Categorical(pz_z[zt] if t > 0 else pz.expand(BSZ, C))
                zt = pyro.sample(
                    f"z_{t}", dzt,
                    infer = {"enumerate": enum_style}
                )
                zs.append(zt)
                if observe:
                    dxt = ds.Categorical(px_z[zt])
                    xt = pyro.sample(f"x_{t}", dxt, obs = data[ind, t])


def optimize_direct(model, data, lr=1e-3):
    "Optimize by marginalizing over the latent variables. (4.1)"
    pyro.clear_param_store()
    optimizer = pyro.optim.Adam({"lr": lr})
    svi = pyro.infer.SVI(
        model.model, model.guide, optimizer, 
        loss = pyro.infer.TraceEnum_ELBO(
            max_iarange_nesting=2, 
            strict_enumeration_warning=True,
        ),
    )

    losses = []
    for epoch in range(1000):
        loss = (svi.step(train_iter.text.transpose(0, 1)) / 
            (train_iter.text.shape[1] * train_iter.text.shape[0]))
        logging.info(f"{epoch}\t{loss}")
        losses.append(loss)
    return losses
        
#models = [(HMM(config), 1e-1), (NaiveBayes(config), 1e-1),  RNNMixture(config)]
#models = [(NaiveBayes(*config), 1e-1)]
models = [(HMM(*config), 1e-1)]

traces = []
for m, lr in models:
    pyro.clear_param_store()
    logging.info(m)
    losses = optimize_direct(m, data, lr)
    
