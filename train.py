from configures import data_args, train_args, model_args
from load_dataset import get_dataloader, balance_dataset, InputDataset, get_cross_dataloader
from DMon import DMon
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import os
import torch
import shutil
from sklearn.model_selection import StratifiedKFold
import time
import pandas as pd


def train():
    print('start loading data====================')
    dataloader = get_dataloader(data_args)

    print('start training model==================')
    gnnNets = DMon(data_args, model_args)
    gnnNets.to('cpu')

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(gnnNets.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

    best_acc = 0.0

    # save path for model
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir(os.path.join('checkpoint', data_args.dataset_name)):
        os.mkdir(os.path.join('checkpoint', f"{data_args.dataset_name}"))
    ckpt_dir = f"./checkpoint/{data_args.dataset_name}/"

    early_stop_count = 0
    for epoch in range(train_args.max_epochs):
        acc = []
        loss_list = []
        gnnNets.train()


        for data in dataloader['train']:
            data = data.to(model_args.device)

            pre, _ = gnnNets(data.x, data.edge_index, data.batch)
            loss = criterion(pre, data.y)

            # optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(gnnNets.parameters(), clip_value=2.0)
            optimizer.step()

            ## record
            _, prediction = torch.max(pre, -1)
            acc.append(prediction.eq(data.y).cpu().numpy())
            loss_list.append(loss.item())

        # report train msg
        epoch_acc = np.concatenate(acc, axis=0).mean()
        epoch_loss = np.average(loss_list)
        print(f"Train Epoch:{epoch}  |Loss: {epoch_loss:.3f} | Train Acc: {epoch_acc:.3f}")

        # report eval msg
        eval_state = evaluate_GC(dataloader['eval'], gnnNets, criterion)
        print(f"Eval Epoch: {epoch} | Loss: {eval_state['loss']:.3f} | Eval Acc: {eval_state['acc']:.3f}")

        # report test msg
        test_state = evaluate_GC(dataloader['test'], gnnNets, criterion)
        print(f"Test Epoch: {epoch} | Loss: {test_state['loss']:.3f} | Test Acc: {test_state['acc']:.3f}")

        # only save the best model
        is_best = (eval_state['acc'] > best_acc)

        if eval_state['acc'] > best_acc:
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count > train_args.early_stopping:
            break

        if is_best:
            best_acc = eval_state['acc']
            early_stop_count = 0
        if is_best or epoch % train_args.save_epoch == 0:
            save_best(ckpt_dir, epoch, gnnNets, model_args.model_name, eval_state['acc'], is_best, 666)

    print(f"The best validation accuracy is {best_acc}.")

    # report test msg
    checkpoint = torch.load(os.path.join(ckpt_dir, f'{model_args.model_name}_best_666.pth'))
    gnnNets.update_state_dict(checkpoint['net'])
    test_state = evaluate_GC(dataloader['test'], gnnNets, criterion)
    print(f"Test: | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f}")


def evaluate_GC(eval_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    gnnNets.eval()

    with torch.no_grad():
        for data in eval_dataloader:

            data = data.to(model_args.device)
            pre, _ = gnnNets(data.x, data.edge_index, data.batch)
            loss = criterion(pre, data.y)

            ## record
            _, prediction = torch.max(pre, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(data.y).cpu().numpy())

        eval_state = {'loss': np.average(loss_list),
                      'acc': np.concatenate(acc, axis=0).mean()}

    return eval_state


def save_best(ckpt_dir, epoch, gnnNets, model_name, eval_acc, is_best, k_fold):
    if is_best:
        print('saving best....')
    else:
        print('saving last....')

    gnnNets.to('cpu')
    state = {
        'net': gnnNets.state_dict(),
        'epoch': epoch,
        'acc': eval_acc
    }
    pth_name = f"{model_name}_latest_{str(k_fold)}.pth"
    best_pth_name = f'{model_name}_best_{str(k_fold)}.pth'
    ckpt_path = os.path.join(ckpt_dir, pth_name)
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copy(ckpt_path, os.path.join(ckpt_dir, best_pth_name))
    gnnNets.to('cpu')



def cross_train():

    dataloader_list = get_cross_dataloader(data_args)
    k_fold_acc = []

    for i in range(len(dataloader_list)):

        dataloader = dataloader_list[i]

        print('start training model with fold ', i)
        gnnNets = DMon(data_args, model_args)
        gnnNets.to('cpu')

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(gnnNets.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

        # save path for model
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir(os.path.join('checkpoint', data_args.dataset_name)):
            os.mkdir(os.path.join('checkpoint', f"{data_args.dataset_name}"))
        ckpt_dir = f"./checkpoint/{data_args.dataset_name}/"

        early_stop_count = 0
        best_acc = 0
        for epoch in range(train_args.max_epochs):
            acc = []
            loss_list = []
            gnnNets.train()

            for data in dataloader['train']:
                data = data.to(model_args.device)

                pre, _ = gnnNets(data.x, data.edge_index, data.batch)
                loss = criterion(pre, data.y)

                # optimization
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(gnnNets.parameters(), clip_value=2.0)
                optimizer.step()

                ## record
                _, prediction = torch.max(pre, -1)
                acc.append(prediction.eq(data.y).cpu().numpy())
                loss_list.append(loss.item())

            # report train msg
            epoch_acc = np.concatenate(acc, axis=0).mean()
            epoch_loss = np.average(loss_list)
            print(f"Train Epoch:{epoch}  |Loss: {epoch_loss:.3f} | Train Acc: {epoch_acc:.3f}")

            # report eval msg
            eval_state = evaluate_GC(dataloader['eval'], gnnNets, criterion)
            print(f"Eval Epoch: {epoch} | Loss: {eval_state['loss']:.3f} | Eval Acc: {eval_state['acc']:.3f}")

            # report test msg
            test_state = evaluate_GC(dataloader['test'], gnnNets, criterion)
            print(f"Test Epoch: {epoch} | Loss: {test_state['loss']:.3f} | Test Acc: {test_state['acc']:.3f}")

            # only save the best model
            is_best = (eval_state['acc'] > best_acc)

            if eval_state['acc'] > best_acc:
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count > train_args.early_stopping:
                break

            if is_best:
                best_acc = eval_state['acc']
                early_stop_count = 0
            if is_best or epoch % train_args.save_epoch == 0:
                save_best(ckpt_dir, epoch, gnnNets, model_args.model_name, eval_state['acc'], is_best, i)

        print(f"The best validation accuracy is {best_acc}.")


        # report test msg
        checkpoint = torch.load(os.path.join(ckpt_dir, f'{model_args.model_name}_best_{str(i)}.pth'))
        gnnNets.update_state_dict(checkpoint['net'])
        test_state = evaluate_GC(dataloader['test'], gnnNets, criterion)
        print(f"Test: | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f}")
        k_fold_acc.append(test_state['acc'])
        

    print(k_fold_acc)
    print("average: ", np.average(k_fold_acc))






if __name__ == "__main__":
    cross_train()

    # # report test msg
    # ckpt_dir = f"./checkpoint/{data_args.dataset_name}/"
    # gnnNets = DMon(data_args, model_args)
    # dataset = get_dataset(data_args.dataset_dir)
    # dataloader = get_dataloader(dataset, data_args)
    # criterion = nn.CrossEntropyLoss()
    # checkpoint = torch.load(os.path.join(ckpt_dir, f'{model_args.model_name}_latest.pth'))
    # gnnNets.update_state_dict(checkpoint['net'])
    # test_state = evaluate_GC(dataloader['eval'], gnnNets, criterion)
    # print(f"Test: | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f}")



