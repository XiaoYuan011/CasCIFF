import math
import pickle
from model.CasCIFF import CasCIFF as Model
import torch
import time
from model.metrics import *
import random

seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
data_name = "aps"
task = "5year"
neibor = '2nei'

if task == "1day":
    observation = 3600 * 24 * 1
    pre_times = [24 * 3600 * 32]
    n_time_interval = (observation) // (3 * 60 * 60)
elif task == "2day":
    observation = 3600 * 24 * 2
    pre_times = [24 * 3600 * 32]
    n_time_interval = (observation) // (3 * 60 * 60)
elif task == "0.5h":
    pre_times = [24 * 3600]
    observation = 1800 * 1
    n_time_interval = 6  # 5m
elif task == "1h":
    pre_times = [24 * 3600]
    observation = 1800 * 2
    n_time_interval = 6  # 10m
elif task == "3year":
    observation = 365 * 3
    pre_times = [365 * 20 + 5]
    n_time_interval = (observation) // (3 * 30)  # 3month
elif task == "5year":
    observation = 365 * 5 + 1
    pre_times = [365 * 20 + 5]
    n_time_interval = (observation) // (3 * 30)  # 3month

time_interval = math.ceil((observation) * 1.0 / n_time_interval)
learning_rate = 5e-4
training_iters = 1000 * 3200 + 1
batch_size = 64
weight_decay = 1e-6
n_steps = 100

increase_node = 1
top = 30
data_type = "CasCN_adj_step{}".format(increase_node)

DATA_PATHA = './dataset/evaluate/{}/{}'.format(data_name, task)
Save_Mode = 'CasCIFF_{}_{}'.format(data_name, task)
Save_Mode = './save_model/' + Save_Mode + "_lr{0}_seed{1}_{2}_{3}_increase_node{4}".format(learning_rate, seed,
                                                                                           data_type, neibor,
                                                                                           increase_node) + '.pth'
train_pkl = DATA_PATHA + "/data_train_{}.pkl".format(data_type)
val_pkl = DATA_PATHA + "/data_val_{}.pkl".format(data_type)
test_pkl = DATA_PATHA + "/data_test_{}.pkl".format(data_type)
isordered = '{}_sample50'.format(neibor) + '_top_' + str(top) + '.pkl'
if neibor == '2nei':
    gg_emb_dim = 100
    emb_dim = 82
    z_dim = 64
    squence_length = 2
elif neibor == "1nei":
    gg_emb_dim = 50
    emb_dim = 50
    z_dim = 64
    squence_length = 1
elif neibor == '3nei':
    gg_emb_dim = 150
    emb_dim = 96
    z_dim = 64
    squence_length = 3
elif neibor == '4nei':
    gg_emb_dim = 200
    emb_dim = 128
    z_dim = 64
    squence_length = 4
elif neibor == '5nei':
    gg_emb_dim = 250
    emb_dim = 128
    z_dim = 64
    squence_length = 5
elif neibor == '6nei':
    gg_emb_dim = 300
    emb_dim = 128
    z_dim = 64
    squence_length = 6
global_train_pkl = DATA_PATHA + "/global_step_train_" + isordered
global_val_pkl = DATA_PATHA + "/global_step_val_" + isordered
global_test_pkl = DATA_PATHA + "/global_step_test_" + isordered
if data_name == "twitter":
    # The initial vertices '-1' are not considered
    feat_in = n_steps + 1
else:
    feat_in = n_steps
print("***********CasCIFF-metrics************")
print('lr:', learning_rate)
print('weight_decay:', weight_decay)
print('dataset:', "global_step_train_" + isordered)
print("data path:", DATA_PATHA)
print('Save Model:', Save_Mode)
print('batch_size:', batch_size)
print('observation:', observation)
print("increase_node:", increase_node)
print('time_interval:', time_interval)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_batch(x, L, y, sz, time, n_time_interval, step, batch_size, num_step, gg_emb, gg_label, is_train=True):
    start = step * batch_size % len(x)
    if start + batch_size >= len(x):
        batch_size_temp = len(x) - start
    else:
        batch_size_temp = batch_size
    batch_y = np.zeros(shape=(batch_size_temp, 1))
    batch_x = []
    batch_L = []
    batch_time_interval_index = []
    batch_rnn_index = []
    batch_gg_emb = []
    batch_gg_label = []
    batch_gg_index = []
    batch_time_serise = []
    for i in range(batch_size_temp):
        id = (i + start) % len(x)
        batch_y[i, 0] = y[id]
        batch_L.append(L[id].todense())
        temp_x = []
        temp_gg_index = []
        for m in range(len(x[id])):
            length = len(x[id][m].data)
            temp_gg_index.append(length)
            temp_x.append(x[id][m].todense())
        temp1 = []
        for node in gg_emb[id]:
            temp2 = []
            for squee in node:
                temp_max = max(1, squee[0])
                temp2.append([t / temp_max for t in squee])
            temp1.append(temp2)
        batch_gg_emb.append(torch.as_tensor(np.array(temp1), dtype=torch.float32, device=device))
        batch_gg_label.append(torch.as_tensor(np.array(gg_label[id]), dtype=torch.long, device=device))
        batch_x.append(temp_x)
        batch_gg_index.append(temp_gg_index)
        batch_time_interval_index_sample = []
        temp_time_serise = np.zeros([1, n_steps])
        count = 1
        for j in range(sz[id]):
            temp_time_serise[0, j] = 1. - time[id][j] / observation
            if count % increase_node == 0:
                temp_time = np.zeros(shape=(n_time_interval))
                k = int(math.floor(time[id][j] / time_interval))
                temp_time[k] = 1
                batch_time_interval_index_sample.append(temp_time)
            count += 1
        if sz[id] % increase_node != 0:
            temp_time = np.zeros(shape=(n_time_interval))
            k = int(math.floor(time[id][-1] / time_interval))
            temp_time[k] = 1
            batch_time_interval_index_sample.append(temp_time)
        if len(batch_time_interval_index_sample) < math.ceil(num_step / increase_node):
            for i in range(math.ceil(num_step / increase_node) - len(batch_time_interval_index_sample)):
                temp_time_padding = np.zeros(shape=(n_time_interval))
                batch_time_interval_index_sample.append(temp_time_padding)
        batch_time_interval_index.append(batch_time_interval_index_sample)
        rnn_index_temp = np.zeros(shape=(math.ceil(n_steps / increase_node)))
        rnn_index_temp[:math.ceil(sz[id] / increase_node)] = 1
        batch_rnn_index.append(rnn_index_temp)
        batch_time_serise.append(temp_time_serise)
    if is_train:
        con = list(
            zip(batch_x, batch_L, batch_y, batch_time_interval_index, batch_rnn_index, batch_gg_emb, batch_gg_label,
                batch_gg_index, batch_time_serise))
        random.shuffle(con)
        batch_x, batch_L, batch_y, batch_time_interval_index, batch_rnn_index, batch_gg_emb, batch_gg_label, batch_gg_index, batch_time_serise = zip(
            *con)

    return torch.as_tensor(np.array(batch_x), dtype=torch.float32, device=device), \
           torch.as_tensor(np.array(batch_L), dtype=torch.float32, device=device), \
           torch.as_tensor(np.array(batch_y), dtype=torch.float32, device=device), \
           torch.as_tensor(np.array(batch_time_interval_index), dtype=torch.float32, device=device), \
           torch.as_tensor(np.array(batch_rnn_index), dtype=torch.float32, device=device), \
           batch_gg_emb, \
           batch_gg_label, \
           np.array(batch_gg_index), \
           torch.as_tensor(np.array(batch_time_serise), dtype=torch.float32, device=device)


print("-----------------start evaluate CasCIFF-------------------")
np.set_printoptions(precision=2)
start = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model(increase_node=increase_node, feat_in=feat_in, squence_length=squence_length, input_dim=gg_emb_dim,
              n_time_interval=n_time_interval, emb_dim=emb_dim, z_dim=z_dim).to(device)
state_dict = torch.load(Save_Mode)
model.load_state_dict(state_dict['model'])
print("model paramerters", [x.numel() for x in model.parameters()])
print("model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
id_test, x_test, L_test, y_test, sz_test, time_test, _ = pickle.load(
    open(test_pkl, 'rb'))
global_x_test, global_y_test = pickle.load(open(global_test_pkl, 'rb'))
best_test_loss = 1000
best_test_acc_loss = 1000
best_test_acc = 0
test_loss = []
test_acc = []
test_acc_loss = []
test_truth = []
test_pred = []
preparedata_time = 0
model.eval()
for test_step in range(math.ceil(len(y_test) / batch_size)):
    test_preparedata = time.time()
    batch_x, batch_L, batch_y, batch_time_interval, batch_rnn_index, batch_gg_emb, batch_gg_lable, batch_gg_index, batch_time_serise = get_batch(
        x_test,
        L_test,
        y_test,
        sz_test,
        time_test,
        n_time_interval,
        test_step,
        batch_size,
        n_steps, global_x_test, global_y_test, is_train=False)
    preparedata_time += time.time() - test_preparedata
    with torch.no_grad():
        _, classfier_loss, regression_loss, classfier_acc, pred2 = model(batch_x, batch_L, batch_gg_emb, batch_y,
                                                                         batch_gg_lable, batch_time_interval,
                                                                         batch_rnn_index, batch_time_serise)
        test_loss.append(regression_loss.cpu().detach().numpy())
        test_acc_loss.append(classfier_loss.cpu().detach().numpy())
        test_acc.append(classfier_acc.cpu().detach().numpy())
        preds = pred2.squeeze(-1).cpu().detach().numpy()
        test_y = batch_y.squeeze(-1).cpu().detach().numpy()
        test_pred += preds.tolist()
        test_truth += test_y.tolist()
best_test_loss = np.mean(test_loss)
best_test_acc = np.mean(test_acc)
best_test_acc_loss = np.mean(test_acc_loss)
print("Finished!\n----------------------------------------------------------------")
total_time = time.time() - start
print("Total Time:{},Preparation data Time:{},Training data Time:{}".format(total_time, preparedata_time,
                                                                            total_time - preparedata_time))
print("Test Loss:{:.4f} Acc Loss{:.4f} Acc:{:.2f}%".format(best_test_loss, best_test_acc_loss, best_test_acc * 100.0))
Old_Evaluate(test_pred, test_truth)
New_Evaluate(test_pred, test_truth)
