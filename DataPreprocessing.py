import pickle
import numpy as np
import copy
import random
import torch
from torch.autograd import Variable
import Event

USE_CUDA = torch.cuda.is_available()
'''
92 inches approximately equals 28 meters, length
50 inches approximately equals 15 meters, width
Because the data is not very large, and not obeys Gauss distribution,
I don't use input minus mean and over std deviation to normalize,
but scale the date to 0-1
'''



def evaluate(net, load_weight_path, test_data_path, output_path):
    net.load_state_dict(torch.load(load_weight_path))
    all_list = pickle.load(open(test_data_path, "rb"))
    gt_data=[]
    i=0
    for test_list in all_list:
        input=[]
        for moment in test_list[:-1]:
            if len(moment.players)!=10:
                fill_player(moment, test_list)
            for i in range(10):
                player = moment.players[i]
                input.append(player.x / 92)
                input.append(player.y / 50)
            input.append(moment.ball.x / 92)
            input.append(moment.ball.y / 50)
        pred = test_list[-1]
        for i in range(10):
            if pred.players[i].x==-1 and pred.players[i].y==-1:
                pred_index = i
                break
        input = np.asanyarray(input)
        input = input.reshape(1, 5, 22)
        input = torch.from_numpy(input)
        input = input.float()
        if USE_CUDA:
            input = input.cuda()
            net = net.cuda()
        input= Variable(input)
        out = net(input)
        res= (out[0,2*pred_index].data[0]*92, out[0,2*pred_index+1].data[0]*50)
        i+=1
        gt_data.append(res)
    pickle.dump(gt_data, open(output_path, "wb"),protocol=2)


def fill_player(moment, moment_list):
    last_record = None
    next_record = None
    len_moment = len(moment_list)
    index = moment_list.index(moment)
    for i in range(index):
        if len(moment_list[i].players) == 10:
            last_record = moment_list[i]
    for i in range(index + 1, len_moment):
        if len(moment_list[i].players) == 10:
            next_record = moment_list[i]
            break
    if not (last_record is None or next_record is None):
        player_info = copy.deepcopy(last_record.players[0])
        t1 = last_record.game_clock
        t2 = moment.game_clock
        t3 = next_record.game_clock
        d1 = t2 - t1
        d2 = t3 - t2
        lamda = d1 / d2
        x1 = last_record.players[0].x
        x2 = next_record.players[0].x
        y1 = last_record.players[0].y
        y2 = next_record.players[0].y
        player_info.x = (x1 + lamda * x2) / (1 + lamda)
        player_info.y = (y1 + lamda * y2) / (1 + lamda)
        moment.players.insert(0, player_info)
    elif last_record is None:
        player_info = copy.deepcopy(next_record.players[0])
        moment.players.insert(0, player_info)
    elif next_record is None:
        player_info = copy.deepcopy(last_record.players[0])
        moment.players.insert(0, player_info)


def split_moment(pred_data, test_data, data):
    for i in range(len(pred_data.players)):
        pred_data_copy = copy.deepcopy(pred_data)
        test_data_copy = copy.deepcopy(test_data)
        player = pred_data_copy.players[i]
        ground_truth = (player.x, player.y)
        player.x = -1
        player.y = -1
        test_data_copy.append(pred_data_copy)
        data.append((test_data_copy, ground_truth))
    return data


def get_train_data(give_len=5, stride=1, splitToTen=False):
    filename = "train.p"
    test_list = pickle.load(open(filename, "rb"))
    print("finish read train.p")
    given_length = give_len
    predict_period = 2
    total_len = given_length + predict_period  # default = 7(5+2)
    output = []  # data used for train/test
    gt_data = []  # ground truth data
    data = []
    output_filename = "train_data_len{}_stride{}.p".format(give_len + 2, stride)
    gt_data_filename = "gt_datalen{}_stride{}.p".format(give_len + 2, stride)

    for event in test_list:
        if len(event.moments) > 50:
            duration = int(event.moments[0].game_clock - event.moments[-1].game_clock)  # last time
            if duration < total_len + 4: continue
            game_clock_list = []
            game_clock_s_list = []
            for moment in event.moments:
                if int(moment.game_clock) not in game_clock_s_list:
                    game_clock_s_list.append(int(moment.game_clock))
                    game_clock_list.append(moment.game_clock)
                    if len(moment.players) != 10:
                        fill_player(moment, event.moments)
            data_list = game_clock_list[2:-2]  # remove first and last 2 seconds
            # game_clock_list = list(sorted(set(game_clock_list),reverse=False)) 
            if len(data_list) > 7:  # change 10->7
                for i in range(0, len(data_list[:-7]), stride):
                    index = i
                    index_list = range(index, index + total_len, 1)
                    choosen_time = np.asarray(game_clock_list)[index_list]
                    choosen_time = np.delete(choosen_time, 5)  # remove 6th second data
                    given_seq = choosen_time[:5]
                    predict_seq = choosen_time[-1]
                    test_data = []
                    pred_data = []
                    test_data_time = []
                    for moment in event.moments:
                        if moment.game_clock in list(given_seq) and moment.game_clock not in test_data_time:
                            test_data.append(moment)
                            test_data_time.append(moment.game_clock)
                        elif moment.game_clock == predict_seq:
                            pred_data = moment
                    if splitToTen:
                        data = split_moment(pred_data, test_data, data)
                    else:
                        test_data.append(pred_data)
                        ans = []
                        for i in range(10):
                            ans.append((pred_data.players[i].x, pred_data.players[i].y))
                        data.append((test_data, ans))
    print("preprocessing data finish!")
    pickle.dump(data, open("my_train.p", "wb"), protocol=2)
    return data


def get_set(batch_size, name=None):
    if name is None:
        all_data = get_train_data()
    else:
        all_data = pickle.load(open(name, "rb"))
    random.shuffle(all_data)
    random.shuffle(all_data)
    length = len(all_data)
    train_index = 1700 # appximately 5:1, train-set and dev-set
    print(" total data items:", length)
    train_data = all_data[:train_index]
    test_data = all_data[train_index:]
    length = len(test_data)
    rest = length % batch_size
    rest = 0 - rest
    mult = length // batch_size
    if rest < 0: test_data = test_data[:rest]
    pickle.dump(train_data, open("train_data.p", "wb"), protocol=2)
    pickle.dump(test_data, open("test_data.p", "wb"), protocol=2)
    return train_data, test_data


def get_trainloader(train_data, batch_size=1, name=None):
    train_data = train_data
    random.shuffle(train_data)
    random.shuffle(train_data)
    loader = []
    start = 0
    for start in range(0,1700,batch_size):
        batch = train_data[start: start+batch_size]
        input = []
        target =[]
        for pair in batch:
            give = pair[0]
            gt=pair[1]
            for moment in give[:-1]:
                for i in range(10):
                    player = moment.players[i]
                    input.append(player.x/92)
                    input.append(player.y/50)
                input.append(moment.ball.x/92)
                input.append(moment.ball.y/50)
            for res in gt:
                target.append(res[0]/92)
                target.append(res[1]/50)
        input = np.asanyarray(input)
        input = input.reshape(batch_size,5,22)
        input=torch.from_numpy(input)
        target = np.asarray(target)
        target = target.reshape(batch_size,20)
        target=torch.from_numpy(target)
        input = input.float()
        target = target.float()
        loader.append((input, target))
    return loader


def get_testloader(test_data, batch_size=1):
    loader = []
    start = 0
    for start in range(0,len(test_data),batch_size):
        batch = test_data[start: start+batch_size]
        assert len(batch)==batch_size
        input = []
        target =[]
        for pair in batch:
            give = pair[0]
            gt=pair[1]
            for moment in give[:-1]:
                for i in range(10):
                    player = moment.players[i]
                    input.append(player.x/92)
                    input.append(player.y/50)
                input.append(moment.ball.x/92)
                input.append(moment.ball.y/50)
            for res in gt:
                target.append(res[0]/92)
                target.append(res[1]/50)
        input = np.asanyarray(input)
        input = input.reshape(batch_size,5,22)
        input=torch.from_numpy(input)
        target = np.asarray(target)
        target = target.reshape(batch_size,20)
        target=torch.from_numpy(target)
        input =input.float()
        target = target.float()
        loader.append((input, target))
    return loader