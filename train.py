from configures import data_args, train_args, model_args
from load_dataset import  get_dataloader, get_dataset
from DMon import DMon
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import os
import torch
import shutil


def train():
    print('start loading data====================')
    dataset = get_dataset(data_args.dataset_dir)
    dataloader = get_dataloader(dataset, data_args)


    print('start training model==================')
    gnnNets = DMon(data_args, model_args)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(gnnNets.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

    best_acc = 0.0
    best_loss = -100.0
    data_size = len(dataset)
    print(f'The total num of dataset is {data_size}')

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
            loss_list.append(loss.item())
            acc.append(prediction.eq(data.y).cpu().numpy())

        # report train msg
        epoch_acc = np.concatenate(acc, axis=0).mean()
        epoch_loss = np.average(loss_list)
        print(f"Train Epoch:{epoch}  |Loss: {epoch_loss:.3f} | Acc: {epoch_acc:.3f}")

        # only save the best model
        is_best = (epoch_acc > best_acc) or (epoch_loss < best_loss and epoch_acc >= best_acc)
        if epoch_acc == best_acc:
            early_stop_count += 1
        if early_stop_count > train_args.early_stopping:
            break
        if is_best:
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                early_stop_count = 0
            if epoch_loss < best_loss:
                best_loss = epoch_loss
        if is_best or epoch % train_args.save_epoch == 0:
            save_best(ckpt_dir, epoch, gnnNets, model_args.model_name, epoch_acc, is_best)

    print(f"The best validation accuracy is {best_acc}.")
    # report test msg
    checkpoint = torch.load(os.path.join(ckpt_dir, f'{model_args.model_name}_best.pth'))
    gnnNets.update_state_dict(checkpoint['net'])
    test_state, _ = test_GC(dataloader['train'], gnnNets, criterion)
    print(f"Test: | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f}")

def test_GC(test_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    predictions = []
    gnnNets.eval()
    with torch.no_grad():
        for data in test_dataloader:
            data = data.to(model_args.device)
            pre, _ = gnnNets(data.x, data.edge_index, data.batch)
            loss = criterion(pre, data.y)

            # record
            _, prediction = torch.max(pre, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(data.y).cpu().numpy())
            predictions.append(prediction)

    test_state = {'loss': np.average(loss_list),
                  'acc': np.average(np.concatenate(acc, axis=0).mean())}


    predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
    return test_state, predictions


def save_best(ckpt_dir, epoch, gnnNets, model_name, eval_acc, is_best):
    print('saving....')
    gnnNets.to('cpu')
    state = {
        'net': gnnNets.state_dict(),
        'epoch': epoch,
        'acc': eval_acc
    }
    pth_name = f"{model_name}_latest.pth"
    best_pth_name = f'{model_name}_best.pth'
    ckpt_path = os.path.join(ckpt_dir, pth_name)
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copy(ckpt_path, os.path.join(ckpt_dir, best_pth_name))


if __name__ == "__main__":
    train()