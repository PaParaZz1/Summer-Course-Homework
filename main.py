import pickle
import argparse
import random
import torch
import torch.nn
import sys
import os
import time
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from DataPreprocessing import get_set,get_testloader,get_trainloader,evaluate
import Model as m
import Event

parser = argparse.ArgumentParser(description='Phase shell arguments')
USE_CUDA = torch.cuda.is_available()
min_loss = 9999999

'''
    I just use a GTX 1080Ti to train model, it is enough for this task
'''


def train(epoch, lr, net, criterion, optimizer, trainloader, batch_size = 100):
    print('\nepoch number: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_id, (inputs, targets) in enumerate(trainloader):
        # CUDA conversion
        if USE_CUDA:
            inputs = inputs.cuda()
            targets =targets.cuda()
            net = net.cuda()
        # calculate loss func
        inputs = Variable(inputs, requires_grad=True)
        targets = Variable(targets, requires_grad=False)
        torch.cuda.empty_cache()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        train_loss = train_loss + float(loss.data[0]) # if use train_loss+=loss, it will make memory overhead
        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("--------------------------\ntrain loss is{}".format(train_loss))


def test(epoch, net, criterion, weight_file_path, testloader):
    net.eval()
    test_loss = 0
    for batch_id, (inputs, targets) in enumerate(testloader):
        # CUDA conversion
        if USE_CUDA:
            inputs = inputs.cuda()
            targets = targets.cuda()
            net.cuda()
        # calculate loss func
        inputs = Variable(inputs, requires_grad=False)
        targets = Variable(targets,requires_grad=False)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.data[0]
    print("--------------------------\ntest loss is{}".format(test_loss))
    return test_loss


def main():
    global min_loss
    # phase argument
    mode = sys.argv[1]
    if len(sys.argv) == 4:
        save_weight_path = sys.argv[2]
        train_epoch = int(sys.argv[3])
    elif len(sys.argv) == 5 :
        load_weight_path = sys.argv[2]
        test_data_path = sys.argv[3]
        output_path = sys.argv[4]
    else:
        print("please input correct command\n")

    
    # Hyperparameter
    input_size = 22 # 10 players and a ball location
    hidden_size = 120 # I try it from 50-150, 120 units and 6 layers is the bset
    num_layers = 6 # I try it from 5-12
    fc_size = 50 # fc_size for two layers fc
    batch_size = 100 # the number of train data is 1700, so I try 25, 50, 100, 170
    learning_rate = 0.002 # init learning rate

    lock_epoch = 1000
    use_history_data = True


    if mode == "train":
        if train_epoch != 1:
            train_epoch = lock_epoch
        if use_history_data:
            train_set= pickle.load(open("train_data.p", "rb"))
            test_set= pickle.load(open("test_data.p", "rb"))
        else:
            train_set, test_set = get_set(batch_size, "my_train.p")

        net = m.lSTM(input_size, hidden_size, num_layers, batch = batch_size, FC_size = fc_size)
        # load pre-trained model for fine-tuning
        if os.path.isfile("weight_00117"):
            net.load_state_dict(torch.load("weight_00117"))
            train_epoch = 100
            learning_rate = 0.00001
        # load data
        start_time = time.time()
        test_loader = get_testloader(test_set, batch_size = batch_size)
        end_time = time.time() - start_time
        print("--------------------------\nload data time: {}min {}s".format(end_time//60, int(end_time%60)))
        # train settings
        criterion = torch.nn.MSELoss() # MSELoss is good for regression
        '''
        I try SGD, Adagrad, Adam, and Adam is the best choice with lr=0.002
        I try lr = 0.5, 0.1, 0.05, 0.01, 0.004, 0.002, 0.001 and select the best one
        no weight_decay will be better in my dev-set loss(a little strange)
        lr adjustment is also important, this multistep param is the best one from which I use
        '''
        optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate) # weight_decay=0.0001 
        lr_scheduler = MultiStepLR(optimizer, milestones = [100, 300, 800], gamma = 0.2)
        # train
        for epoch in range(train_epoch):
            lr_scheduler.step()
            lr = lr_scheduler.get_lr()[0]
            train_loader =get_trainloader(train_set, batch_size = batch_size)
            train(epoch, lr, net, criterion, optimizer, train_loader, batch_size=batch_size)
            cur_loss = test(epoch, net, criterion, save_weight_path, test_loader)
            if cur_loss < min_loss: # select the best model by dev-set loss
                print('save the best model')
                torch.save(net.cpu().state_dict(), save_weight_path)
                min_loss = cur_loss
        print("global min loss:{}".format(min_loss))
    elif mode == "test":
        net = m.lSTM(input_size, hidden_size, num_layers, batch=1, FC_size=fc_size)
        evaluate(net, load_weight_path, test_data_path, output_path)


if __name__ == "__main__":
   main()


