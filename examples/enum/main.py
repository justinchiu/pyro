
import logging

import torch
from torch.distributions import constraints

import torchtext.data as data

import pyro
import pyro.distributions as ds
from pyro import poutine
from pyro.infer import TraceEnum_ELBO
import pyro.optim

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)
#pyro.enable_validation(True)

if True:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

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


class NaiveBayes:
    def __init__(self, V, C):
        self.V = V
        self.C = C

    def model(self, data):
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
            """
            with pyro.iarange("T", T) as t:
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
        self.batch_x = True

    def model(self, data):
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
                if not self.batch_x:
                    dxt = ds.Categorical(px_z[zt])
                    xt = pyro.sample(f"x_{t}", dxt, obs = data[ind, t])
            if self.batch_x:
                z = torch.stack(zs) # T x N
                with pyro.iarange("T", T) as t:
                    dxt = ds.Categorical(px_z[z])
                    xt = pyro.sample(f"x", dxt, obs = data[ind].t())


def no_guide(*args):
    pass


def optimize_direct(model, data, lr=1e-3):
    "Optimize by marginalizing over the latent variables. (4.1)"
    pyro.clear_param_store()
    optimizer = pyro.optim.Adam({"lr": lr})
    svi = pyro.infer.SVI(
        model, no_guide, optimizer, 
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
    losses = optimize_direct(m.model, data, lr)
    
