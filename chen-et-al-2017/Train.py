from torch.autograd import Variable
import pickle as pkl
import numpy
import torch
import torch.nn as nn
import logging

from NLI_Model import NLI
from Utils import load_data
from Utils import load_data_for_batch_processing
from Utils import load_checkpoint
from Utils import save_checkpoint
from Settings import *

logging.basicConfig(filename=log_file_name, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def convertToVariables(x1, x1_mask, char_x1, char_x1_mask, x2, x2_mask, char_x2, char_x2_mask, l):
    return Variable(x1), Variable(x1_mask), Variable(char_x1), Variable(char_x1_mask), Variable(x2), Variable(
        x2_mask), Variable(char_x2), Variable(char_x2_mask), Variable(l)


with open('word_sequence/vocab_cased.pkl', 'rb') as f:
    worddicts = pkl.load(f)
worddicts_r = dict()

for kk, vv in worddicts.items():
    worddicts_r[vv] = kk
if batch_processing:
    train, valid, valid_out_domain = load_data_for_batch_processing(train_file_loc, dev_file_loc, valid_out_domain_file_loc, new_batch_size)
else:
    train, valid, valid_out_domain = load_data(train_file_loc, dev_file_loc, valid_out_domain_file_loc, batch_size)

model = NLI(dim_word, char_nout, dim_char_emb, 'data/glove.840B.300d.txt', worddicts, num_words, dim_hidden).cuda()
print(print_model_load)
logger.info(print_model_load)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(print_optimizer)
logger.info(print_optimizer)


def pred_acc(iterator):
    model.eval()
    valid_acc = 0
    n_done = 0
    validation_loss = 0
    loss_n = nn.CrossEntropyLoss(size_average=False)
    for x1, x1_mask, char_x1, char_x1_mask, x2, x2_mask, char_x2, char_x2_mask, l in iterator:
        n_done += x1.size(1)
        premise, premise_mask, char_premise, char_premise_mask, hypothesis, hypothesis_mask, char_hypothesis, char_hypothesis_mask, l = convertToVariables(
            x1, x1_mask, char_x1, char_x1_mask, x2, x2_mask, char_x2, char_x2_mask, l)
        outputs = model(premise, char_premise, premise_mask, char_premise_mask, hypothesis, char_hypothesis,
                        hypothesis_mask, char_hypothesis_mask, l)
        valid_acc += (outputs.max(1)[1] == l).sum().data[0]
        validation_loss += loss_n(outputs, l).data[0]

    valid_acc = 1.0 * valid_acc / n_done
    validation_loss = 1.0 * validation_loss / n_done
    return valid_acc, validation_loss


if reload:
    checkpoint = load_checkpoint(model_path)
    if checkpoint is not None:
        print(print_reload)
        logger.info(print_reload)
        last_exec = checkpoint[const_last_exec]
        model.load_state_dict(checkpoint[const_model_state_dict])
        best_acc = checkpoint[const_best_val_acc]
        optimizer.load_state_dict(checkpoint[const_optimizer_state_dict])

        train_acc_history = checkpoint[const_train_acc_history]
        valid_acc_history = checkpoint[const_valid_acc_history]
        valid_out_domain_acc_history = checkpoint[const_valid_out_domain_acc_history]

        train_loss_history = checkpoint[const_train_loss_history]
        valid_loss_history = checkpoint[const_valid_loss_history]
        valid_out_domain_loss_history = checkpoint[const_valid_out_domain_loss_history]

        wait_counter = checkpoint[const_wait_counter]
        print("Loaded valid accuracy history = ", valid_acc_history)
        print("Loaded valid_out_domain accuracy history = ", valid_out_domain_acc_history)
        print("Loaded best valid accuracy history = ", best_acc)
        print("Loaded last executed training data index = ", last_exec)
    else:
        print(print_no_reload)
        logger.info(print_no_reload)
        last_exec = 0
        best_acc = 0.0

        train_acc_history = []
        valid_acc_history = []
        valid_out_domain_acc_history = []

        train_loss_history = []
        valid_loss_history = []
        valid_out_domain_loss_history = []

        wait_counter = 0

else:
    print(print_no_reload)
    logger.info(print_no_reload)
    last_exec = 0
    best_acc = 0.0
    train_acc_history = []
    valid_acc_history = []
    valid_out_domain_acc_history = []

    train_loss_history = []
    valid_loss_history = []
    valid_out_domain_loss_history = []
    wait_counter = 0

loss = nn.CrossEntropyLoss(size_average=False)
uidx = 0
bad_counter = 0
end = False

print(print_training)
for epoch in range(num_epochs):
    train_loss = 0
    n_train_done = 0
    train_acc = 0
    for x1, x1_mask, char_x1, char_x1_mask, x2, x2_mask, char_x2, char_x2_mask, l in train:
        n_train_done += x1.size(1)
        model.train()
        uidx += 1
        if (uidx < last_exec):
            abc = len(x1[0])
            continue

        premise, premise_mask, char_premise, char_premise_mask, hypothesis, hypothesis_mask, char_hypothesis, char_hypothesis_mask, l = convertToVariables(
            x1, x1_mask, char_x1, char_x1_mask, x2, x2_mask, char_x2, char_x2_mask, l)

        model.zero_grad()
        optimizer.zero_grad()

        outputs = model(premise, char_premise, premise_mask, char_premise_mask, hypothesis, char_hypothesis,
                        hypothesis_mask, char_hypothesis_mask, l)

        lossy = loss(outputs, l)
        train_loss += lossy.data[0]
        train_acc += (outputs.max(1)[1] == l).sum().data[0]
        lossy.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), 10.0)

        optimizer.step()
        if (uidx % 20 == 0):
            print_stmt_uidx = str.format("uidx = {0}, Loss = {1}", uidx, lossy.data[0])
            print(print_stmt_uidx)
            logger.info(print_stmt_uidx)

        if uidx % eval_period == 0:
            valid_acc, valid_loss = pred_acc(valid)
            valid_out_domain_acc, valid_out_domain_loss = pred_acc(valid_out_domain)
            print_stmnt_eval = str.format(
                "Epoch = {0} ,  Valid Accuracy = {1}, valid_out_domain Accuracy = {2}, Valid Loss = {3}, valid_out_domain Loss = {4}", epoch,
                valid_acc, valid_out_domain_acc, valid_loss, valid_out_domain_loss)
            print(print_stmnt_eval)
            logger.info(print_stmnt_eval)

            valid_acc_history.append(valid_acc)
            valid_out_domain_acc_history.append(valid_out_domain_acc)
            valid_loss_history.append(valid_loss)
            valid_out_domain_loss_history.append(valid_out_domain_loss)

            if valid_acc < numpy.array(valid_acc_history).max():
                wait_counter += 1

            if wait_counter >= waiting_period:
                bad_counter += 1
                wait_counter = 0

            if bad_counter > patience:
                print('Early Stopped. Best Model saved at ' + model_path)
                end = True
                break

            if (valid_acc > best_acc):
                print('saving model')
                best_acc = valid_acc
                save_checkpoint({const_last_exec: uidx,
                                 const_epoch: epoch,
                                 const_best_val_acc: valid_acc,
                                 const_model_state_dict: model.state_dict(),
                                 const_optimizer_state_dict: optimizer.state_dict(),
                                 const_train_acc_history: train_acc_history,
                                 const_valid_acc_history: valid_acc_history,
                                 const_valid_out_domain_acc_history: valid_out_domain_acc_history,
                                 const_train_loss_history: train_loss_history,
                                 const_valid_loss_history: valid_loss_history,
                                 const_valid_out_domain_loss_history: valid_out_domain_loss_history,
                                 const_wait_counter: wait_counter}, model_path)
    
    if(uidx<last_exec):
        continue
    train_acc = 1.0 * train_acc / n_train_done
    train_loss = 1.0 * train_loss / n_train_done
    valid_acc, valid_loss = pred_acc(valid)
    valid_out_domain_acc, valid_out_domain_loss = pred_acc(valid_out_domain)
    train_acc_history.append(train_acc)

    print_stmnt_eval = str.format(
        "Completed Epoch = {0} , Train Accuracy = {1}, Train Loss = {2} Valid Accuracy = {3}, valid_out_domain Accuracy = {4}, Valid Loss = {5}, valid_out_domain Loss = {6}",
        (epoch + 1), train_acc, train_loss, valid_acc, valid_out_domain_acc, valid_loss, valid_out_domain_loss)
    print(print_stmnt_eval)
    logger.info(print_stmnt_eval)

    if end:
        break
