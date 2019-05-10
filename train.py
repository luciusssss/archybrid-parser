#coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import json
from io import open
import os
import time
import random

from bilstm import *
from archybrid import *

def train(model, optimizer, sentences):
    epoch_loss = 0.0
    epoch_arc_error = 0
    epoch_arc_cnt = 0
    sentence_cnt = 0

    random.shuffle(sentences)

    model.train()

    for sentence in sentences:
        
        optimizer.zero_grad()

        mlosses, mloss, err = model(sentence)

        epoch_loss += mloss
        epoch_arc_error += err 
        epoch_arc_cnt += len(sentence)

        if len(mlosses):
            mlosses = sum(mlosses)
            mlosses.backward()
            optimizer.step()
        ##############
        # Here we can use footnote 8 in Eli's original paper
        ##############

        sentence_cnt += 1
        # if sentence_cnt % 50 == 0:
        #     print("sentcnt:", sentence_cnt, "mloss:", epoch_loss, "err:", epoch_arc_error)

    return epoch_loss / epoch_arc_cnt, epoch_arc_error/epoch_arc_cnt

def evaluate(model, sentences):
    model.eval()

    eval_loss = 0.0
    eval_arc_error = 0
    eval_arc_cnt = 0

    sent_cnt = 0
    for sentence in sentences:
        mloss, err_cnt, config = model(sentence)

        eval_loss += mloss
        eval_arc_error += err_cnt
        eval_arc_cnt += len(sentence)

        sent_cnt += 1
        #if sent_cnt % 100 == 0:
            #print("eval: sent cnt:", sent_cnt)
            #print("Error cnt:", eval_arc_error, "Arc cnt:", eval_arc_cnt)
    return eval_loss / eval_arc_cnt, eval_arc_error/eval_arc_cnt



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



# Ensure reproducibility
SEED = 2019
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

POS_DIC = {'AD': 0, 'AS': 1, 'BA': 2, 'CC': 3, 'CD': 4, 'CS': 5, 'DEC': 6, 'DEG': 7, 'DER': 8, 'DEV': 9, 'DT': 10, 'ETC': 11, 'FW': 12, 'JJ': 13, 'LB': 14, 'LC': 15, 'M': 16, 'MSP': 17, 'NN': 18, 'NR': 19, 'NT': 20, 'OD': 21, 'P': 22, 'PN': 23, 'PU': 24, 'SB': 25, 'SP': 26, 'VA': 27, 'VC': 28, 'VE': 29, 'VV': 30}
# Hyper parameters
BATCH_SIZE = 16
EMB_DIM = 300
POS_DIM = 32
HIDDEN_DIM = 200
HIDDEN_DIM2 = 100
N_LAYERS = 2
DROPOUT = 0.5

with open('train_data.json', "r", encoding = "utf-8") as f:
    training_data = json.loads(f.read())

with open('dev_data.json', "r", encoding = "utf-8") as f:
    dev_data = json.loads(f.read())

with open('test_data.json', "r", encoding = "utf-8") as f:
    test_data = json.loads(f.read())

model = TransitionBasedDependencyParsing(POS_DIC, EMB_DIM, POS_DIM, HIDDEN_DIM, HIDDEN_DIM2, N_LAYERS, DROPOUT)
optimizer = optim.Adam(model.parameters())
device = torch.device('cuda')
model = model.to(device)


N_EPOCHS = 10
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'model2_new_dataset.pt')
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)

best_dev_error = 1.0
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
eval_loss, best_dev_error = evaluate(model, dev_data)
print(best_dev_error)

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss, train_error = train(model, optimizer, training_data)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print('Epoch:', epoch+1, '| Time:', epoch_mins, 'm', epoch_secs, 's', 'Train Loss:', train_loss)
    
    #eval_loss, eval_error = evaluate(model, training_data)
    #print("Train Acc:", 1-eval_error)
    eval_loss, eval_error = evaluate(model, dev_data)
    print("Dev Acc:", 1-eval_error)
    if eval_error < best_dev_error:
        best_dev_error = eval_error
        torch.save(model.state_dict(), MODEL_SAVE_PATH)



model.load_state_dict(torch.load(MODEL_SAVE_PATH))
eval_loss, eval_error = evaluate(model, test_data)
print("Eval Loss:", eval_loss, "Accuracy:", 1 - eval_error)
