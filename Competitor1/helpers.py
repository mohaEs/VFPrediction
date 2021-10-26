#!/usr/bin/env python
import numpy as np
import keras
import json, glob, random


class LossHistory(keras.callbacks.Callback):
    def __init__(self, dataset, rundir, rootdir="../runs/"):
      super(LossHistory, self).__init__()
      self.dataset = dataset
      self.rootdir = rootdir
      self.rundir = rundir

    def on_train_begin(self, logs={}):
        self.lastiter = 0

    def on_epoch_end(self, batch, logs={}):
        self.lastiter += 1
        with open("%s/%s/history.txt" % (self.rootdir, self.rundir), "a") as fout:
                fout.write("%s\t%s\ttrain\t%d\t%.8f\n" % ("mse", self.dataset, self.lastiter, logs.get("loss")))
                #fout.write("%s\t%s\ttrain\t%d\t%.8f\n" % ("rmse", self.dataset, self.lastiter, logs.get("rmse")))
                fout.write("%s\t%s\tvalid\t%d\t%.8f\n" % ("mse", self.dataset, self.lastiter, logs.get("val_loss")))
                #fout.write("%s\t%s\tvalid\t%d\t%.8f\n" % ("rmse", self.dataset, self.lastiter, logs.get("val_rmse")))

    def on_batch_end(self, batch, logs={}):
        self.lastiter += 1
        with open("%s/%s/history.txt" % (self.rootdir, self.rundir), "a") as fout:
                fout.write("%s\t%s\ttrain\t%d\t%.8f\n" % ("mse", self.dataset, self.lastiter, logs.get("loss")))
                #fout.write("%s\t%s\ttrain\t%d\t%.8f\n" % ("rmse", self.dataset, self.lastiter, logs.get("rmse")))


class Dataset:
  def __init__(self, datagrp, bs, datasets, lyr, ryr, rep, randomize = True, onlyoneyear=True):
    self.datagrp = datagrp
    self.bs = bs
    self.randomize = randomize
    ptgrps = json.load(open("ptgroups-%s.json" % rep))
    exclude = {}
    with open("excludetest.txt") as fin:
      for l in fin:
        exclude[l.strip()] = 1
    alldata = []
    for f in glob.glob("../pairs/*.json"):
      if datagrp == "test" and exclude.has_key(f):
        continue
      ptid = f.split("/")[-1].split("-")[0]
      if datagrp != ptgrps[ptid]:
        continue
      data = json.load(open(f))
      if onlyoneyear and not (data["delta"] >= lyr and data["delta"] < ryr):
        continue
      data["deltac"] = round(data["delta"])
      target = np.asarray(data["rd"], dtype=np.float)
      inputn = np.asarray(data["ld"], dtype=np.float)
      target = np.reshape(target, (target.shape[0], target.shape[1], 1))
      inputn = np.reshape(inputn, (inputn.shape[0], inputn.shape[1], 1))
      for d in datasets:
        aux = self.__encodeVar(d, target.shape[0], target.shape[1], data[d])
        inputn = np.concatenate((inputn, aux), axis = 2)
      data["x"] = inputn
      data["y"] = target
      data["ptid"] = ptid
      alldata.append(data)
    self.alldata = alldata
    self.datasets = datasets

  def __encodeVar(self, var, x1, x2, val):
    mapping = {	"age": 	("cont", None), 
		"gender": ("cat", ("M", "F")),
		"deltac": ("cat", (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)),
		"delta": ("cont", None),
		"testnum": ("cont", None),
		"eye": ("cat", ("R", "L")) }
    if mapping[var][0] == "cat":
      return self.__encodeCategorical(var, x1, x2, mapping[var][1], val)
    elif mapping[var][0] == "cont":
      return self.__encodeContinuous(var, x1, x2, val)

  def __encodeCategorical(self, var, x1, x2, values, val):
    out = np.zeros((x1, x2, len(values)), dtype=np.float)
    for i, d in enumerate(values):
      if d == val:
        out[:, :, i] = 1.
    return out
    

  def __encodeContinuous(self, var, x1, x2, val):
    out = np.zeros((x1, x2, 1), dtype=np.float)
    out[:, :, 0] = 1. * val
    return out

  def getdatashape(self):
    return self.alldata[0]["x"].shape

  def getIterPerEpoch(self):
    if len(self.alldata) % self.bs == 0:
      return int(len(self.alldata) / self.bs)
    return int(len(self.alldata) / self.bs) + 1

  def loader(self, debug=False):
    (x1, x2, x3) = self.getdatashape()
    l = len(self.alldata)
    x = np.zeros((l, x1, x2, x3), dtype=np.float)
    y = np.zeros((l, x1, x2, 1), dtype=np.float)
    for i in range(l):
      x[i, :, :, :] = np.copy(self.alldata[i]["x"])
      y[i, :, :, :] = np.copy(self.alldata[i]["y"])
    if not debug:
      return x, y
    return self.alldata


  def loaderold(self, debug=False, noiseGen=False):
    p = list(range(len(self.alldata)))
    (x1, x2, x3) = self.getdatashape()
    while True:
      if self.randomize:
        random.shuffle(p)
      curi = 0
      batch_x = np.zeros((self.bs, x1, x2, x3), dtype=np.float)
      batch_y = np.zeros((self.bs, x1, x2, 1), dtype=np.float)
      for i in p:
        if noiseGen:
          noise = 1.0 * (batch_x[curi, :, :, 0] > 0) *  np.random.normal(20.0, 5, (x1, x2))
          batch_x[curi, :, :, 0]  = noise
        else:
          batch_x[curi, :, :, :]  = np.copy(self.alldata[i]["x"])
        batch_y[curi, :, :, :]  = np.copy(self.alldata[i]["y"])
        if self.randomize:
          noise = 1.0 * (batch_x[curi, :, :, 0] > 0) *  np.random.normal(0.0, 0.5, (x1, x2))
          batch_x[curi, :, :, 0] = np.round(batch_x[curi, :, :, 0] + noise,2)
        curi += 1
        if curi == self.bs:
          if debug:
            yield(batch_x, batch_y, self.alldata[i])
          else:
            yield(batch_x, batch_y)
          curi = 0
          batch_x = np.zeros((self.bs, x1, x2, x3), dtype=np.float)
          batch_y = np.zeros((self.bs, x1, x2, 1), dtype=np.float)
      if curi != 0:
        batch_x = batch_x[0:curi, :, :, :]
        batch_y = batch_y[0:curi, :, :, :]
        yield(batch_x, batch_y)
  
if __name__ == "__main__":
  auxdata = ["delta", "age", "gender", "testnum"]
  dataset = Dataset("valid", 32, auxdata)
  for x1, y1 in dataset.loader():
    print x1
    break
