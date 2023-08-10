import math
import  pickle
from model.CasCIFF import CasCIFF as Model
import torch
import time
from log.Logger import Logger
import sys
from model.metrics import *
from torch.utils.tensorboard import SummaryWriter
import random

seed =1
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
data_name="weibo"
task="1h"
neibor='2nei'

if task=="1day":
    observation = 3600 * 24 * 1
    pre_times = [24 * 3600 * 32]
    n_time_interval = (observation) // (3 * 60 * 60)
elif task=="2day":
    observation = 3600 * 24 * 2
    pre_times = [24 * 3600 * 32]
    n_time_interval = (observation) // (3 * 60 * 60)
elif task=="0.5h":
    pre_times = [24 * 3600]
    observation =1800*1
    n_time_interval = 6  # 5m
elif task=="1h":
    pre_times = [24 * 3600]
    observation =1800*2
    n_time_interval = 6  # 10m
elif task=="3year":
    observation =365*3
    pre_times = [365*20+5]
    n_time_interval = (observation)//(3*30)  #3month
elif task=="5year":
    observation = 365*5+1
    pre_times = [365 * 20 + 5]
    n_time_interval = (observation)//(3*30)  #3month

time_interval = math.ceil((observation) * 1.0 / n_time_interval)
learning_rate = 5e-4
training_iters = 1000 * 3200 + 1
batch_size = 64
weight_decay = 1e-6
n_steps = 100

increase_node=1
top = 30
data_type="CasCN_adj_step{}".format(increase_node)


DATA_PATHA = './dataset/CasCIFF/{}/{}'.format(data_name,task)
Save_Mode = 'CasCIFF_{}_{}_{}'.format(data_name,task,neibor)
sys.stdout = Logger('./log/' + Save_Mode + '.txt')
writer = SummaryWriter('./runs/{}/{}/CasCIFF'.format(data_name,task)+'_lr{0}_seed{1}_{2}_increase_node{3}'.format(learning_rate, seed,neibor,increase_node))
Save_Mode = './save_model/' + Save_Mode+"_lr{0}_seed{1}_{2}_increase_node{3}".format(learning_rate, seed,data_type,increase_node) + '.pth'
train_pkl = DATA_PATHA + "/data_train_{}.pkl".format(data_type)
val_pkl = DATA_PATHA + "/data_val_{}.pkl".format(data_type)
test_pkl = DATA_PATHA + "/data_test_{}.pkl".format(data_type)
isordered = '{}_sample50'.format(neibor) + '_top_' + str(top) + '.pkl'
if neibor == '2nei':
    gg_emb_dim = 100
    emb_dim = 82
    z_dim = 64
    squence_length = 2
elif neibor =="1nei":
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
if data_name=="twitter":
    # The initial vertices '-1' are not considered
    feat_in=n_steps+1
else:
    feat_in=n_steps
print("***********CasCIFF************")
print('seed:', seed)
print('lr:', learning_rate)
print('weight_decay:', weight_decay)
print('dataset:', "global_step_train_" + isordered)
print("data path:",DATA_PATHA)
print('Save Model:',Save_Mode)
print('batch_size:',batch_size)
print('observation:',observation)
print("increase_node:",increase_node)
print('time_interval:',time_interval)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_batch(x, L, y, sz, time, n_time_interval, step, batch_size, num_step,gg_emb,gg_label,is_train=True):
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
    batch_gg_emb=[]
    batch_gg_label=[]
    batch_gg_index=[]
    batch_time_serise=[]
    for i in range(batch_size_temp):
        id = (i + start) % len(x)
        batch_y[i, 0] = y[id]
        batch_L.append(L[id].todense())
        temp_x = []
        temp_gg_index=[]
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
        count=1
        for j in range(sz[id]):
            temp_time_serise[0, j] = 1. - time[id][j] / observation
            if count%increase_node==0:
                temp_time = np.zeros(shape=(n_time_interval))
                k = int(math.floor(time[id][j] / time_interval))
                temp_time[k] = 1
                batch_time_interval_index_sample.append(temp_time)
            count += 1
        if sz[id]%increase_node!=0:
            temp_time = np.zeros(shape=(n_time_interval))
            k = int(math.floor(time[id][-1] / time_interval))
            temp_time[k] = 1
            batch_time_interval_index_sample.append(temp_time)
        if len(batch_time_interval_index_sample) < math.ceil(num_step/increase_node):
            for i in range(math.ceil(num_step/increase_node) - len(batch_time_interval_index_sample)):
                temp_time_padding = np.zeros(shape=(n_time_interval))
                batch_time_interval_index_sample.append(temp_time_padding)
        batch_time_interval_index.append(batch_time_interval_index_sample)
        rnn_index_temp = np.zeros(shape=(math.ceil(n_steps / increase_node)))
        rnn_index_temp[:math.ceil(sz[id] / increase_node)] = 1
        batch_rnn_index.append(rnn_index_temp)
        batch_time_serise.append(temp_time_serise)
    if is_train:
        con = list(zip(batch_x, batch_L,batch_y,batch_time_interval_index,batch_rnn_index,batch_gg_emb,batch_gg_label,batch_gg_index,batch_time_serise))
        random.shuffle(con)
        batch_x, batch_L,batch_y,batch_time_interval_index,batch_rnn_index,batch_gg_emb,batch_gg_label,batch_gg_index,batch_time_serise = zip(*con)

    return torch.as_tensor(np.array(batch_x), dtype=torch.float32, device=device), \
           torch.as_tensor(np.array(batch_L), dtype=torch.float32, device=device), \
           torch.as_tensor(np.array(batch_y), dtype=torch.float32, device=device), \
           torch.as_tensor(np.array(batch_time_interval_index), dtype=torch.float32, device=device), \
           torch.as_tensor(np.array(batch_rnn_index), dtype=torch.float32, device=device), \
           batch_gg_emb, \
           batch_gg_label, \
           np.array(batch_gg_index), \
           torch.as_tensor(np.array(batch_time_serise), dtype=torch.float32, device=device)

id_train, x_train, L, y_train, sz_train, time_train, vocabulary_size = pickle.load(
    open(train_pkl, 'rb'))
id_val, x_val, L_val, y_val, sz_val, time_val, _ = pickle.load(open(val_pkl, 'rb'))
global_x_train,global_y_train=pickle.load(open(global_train_pkl,'rb'))
global_x_val,global_y_val=pickle.load(open(global_val_pkl,'rb'))
display_step = max(100, math.ceil(len(sz_train) / batch_size))

print("-----------------display step-------------------")
print("display step:" + str(display_step))

# determine the way floating point numbers,arrays and other numpy object are displayed
np.set_printoptions(precision=2)
start = time.time()
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model(increase_node=increase_node,feat_in=feat_in,squence_length=squence_length,input_dim=gg_emb_dim,n_time_interval=n_time_interval,emb_dim=emb_dim,z_dim=z_dim).to(device)

print("model paramerters", [x.numel() for x in model.parameters()])
print("model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
step = 0
best_val_loss = 1000
best_test_loss = 1000
best_test_acc_loss = 1000
best_val_acc_loss = 1000
best_val_acc = 0
best_test_acc = 0
max_try = 10
patience = max_try
# optimizer = ScheduledOptim(
#         torch.optim.Adam(
#             model.parameters(),
#             betas=(0.9, 0.98), eps=1e-09),
#         64, 1000)
optimizer = torch.optim.Adam(model.parameters(),
                                          lr=learning_rate,
                                          weight_decay=weight_decay)
train_loss = []
train_acc_loss = []
train_acc = []
model.train()
preparedata_time=0
relative_time=time.time()
while step * batch_size < training_iters:
    train_preparedata=time.time()
    batch_x, batch_L, batch_y, batch_time_interval, batch_rnn_index,batch_gg_emb,batch_gg_lable,batch_gg_index,batch_time_serise = get_batch(
                                                                                x_train,
                                                                                L,
                                                                                y_train,
                                                                                sz_train,
                                                                                time_train,
                                                                                n_time_interval,
                                                                                step,
                                                                                batch_size,
                                                                                n_steps,global_x_train,global_y_train)
    preparedata_time+=time.time()-train_preparedata
    optimizer.zero_grad(set_to_none=True)
    loss,classfier_loss,regression_loss,classfier_acc,pred2=model(batch_x,batch_L,batch_gg_emb,batch_y,batch_gg_lable,batch_time_interval,batch_rnn_index,batch_time_serise) #batch_x,batch_l,batch_global_x,batch_global_index,batch_y,batch_label
    loss.backward()
    optimizer.step()
    # optimizer.update_learning_rate()
    train_loss.append(regression_loss.item())
    train_acc_loss.append(classfier_loss.item())
    train_acc.append(classfier_acc.item())
    step += 1
    if step % display_step == 0:
        val_loss = []
        val_acc = []
        val_acc_loss = []
        model.eval()
        for val_step in range(math.ceil(len(y_val) / batch_size)):
            val_preparedata = time.time()
            batch_x, batch_L, batch_y, batch_time_interval, batch_rnn_index, batch_gg_emb, batch_gg_lable, batch_gg_index,batch_time_serise = get_batch(
                x_val,
                L_val,
                y_val,
                sz_val,
                time_val,
                n_time_interval,
                val_step,
                batch_size,
                n_steps, global_x_val, global_y_val,is_train=False)
            preparedata_time+=time.time()-val_preparedata
            with torch.no_grad():
                _,classfier_loss,regression_loss,classfier_acc,pred2=model(batch_x,batch_L,batch_gg_emb,batch_y,batch_gg_lable,batch_time_interval,batch_rnn_index,batch_time_serise)
                val_loss.append(regression_loss.item())
                val_acc_loss.append(classfier_loss.item())
                val_acc.append(classfier_acc.item())
        if np.mean(val_loss) < best_val_loss:
            best_val_loss = np.mean(val_loss)
            best_val_acc = np.mean(val_acc)
            patience = max_try
            best_val_acc_loss = np.mean(val_acc_loss)
            torch.save({'model': model.state_dict()}, Save_Mode)
        print("#" + str(step / display_step) +
              ", Training Loss= " + "{:.4f}".format(np.mean(train_loss)) +
              ", Training Acc Loss= " + "{:.4f}".format(np.mean(train_acc_loss)) +
              ", Training Acc= " + "{:.2f}%".format(np.mean(train_acc) * 100.0) +
              ", Validation Loss= " + "{:.4f}".format(np.mean(val_loss)) +
              ", Validation Acc Loss= " + "{:.4f}".format(np.mean(val_acc_loss)) +
              ", Validation Acc= " + "{:.2f}%".format(np.mean(val_acc) * 100.0) +
              ", Best Valid Loss= " + "{:.4f}".format(best_val_loss) +
              ", Best Valid Acc Loss= " + "{:.4f}".format(best_val_acc_loss) +
              ", Best Valid Acc= " + "{:.2f}%".format(best_val_acc * 100.0) +
              ", Patience= " + "{:.2f}".format(patience) +
              ", IterTime=" + "{:.2f}s".format(time.time() - relative_time)
              )
        writer.add_scalars('data/Loss', {'train loss': np.mean(train_loss),
                                         'val loss': np.mean(val_loss)}, (step // display_step))
        writer.add_scalars('data/Acc', {'train acc': np.mean(train_acc),
                                        'val acc': np.mean(val_acc)}, (step // display_step))
        writer.add_scalars('data/Classfier Loss', {'train loss': np.mean(train_acc_loss),
                                                   'val loss': np.mean(val_acc_loss)}, (step // display_step))
        relative_time=time.time()
        train_loss = []
        train_acc_loss = []
        train_acc = []
        model.train()
        patience -= 1
        if not patience:
            break
id_test, x_test, L_test, y_test, sz_test, time_test, _ = pickle.load(
    open(test_pkl, 'rb'))
global_x_test,global_y_test=pickle.load(open(global_test_pkl,'rb'))
test_loss = []
test_acc=[]
test_acc_loss = []
test_truth = []
test_pred = []
state_dict = torch.load(Save_Mode)
model.load_state_dict(state_dict['model'])
model.eval()
for test_step in range(math.ceil(len(y_test) / batch_size)):
    test_preparedata = time.time()
    batch_x, batch_L, batch_y, batch_time_interval, batch_rnn_index, batch_gg_emb, batch_gg_lable, batch_gg_index,batch_time_serise = get_batch(
        x_test,
        L_test,
        y_test,
        sz_test,
        time_test,
        n_time_interval,
        test_step,
        batch_size,
        n_steps, global_x_test, global_y_test,is_train=False)
    preparedata_time+=time.time()-test_preparedata
    with torch.no_grad():
        _,classfier_loss,regression_loss,classfier_acc,pred2=model(batch_x,batch_L,batch_gg_emb,batch_y,batch_gg_lable,batch_time_interval,batch_rnn_index,batch_time_serise)
        test_loss.append(regression_loss.cpu().detach().numpy())
        test_acc_loss.append(classfier_loss.cpu().detach().numpy())
        test_acc.append(classfier_acc.cpu().detach().numpy())
        preds = pred2.squeeze(-1).cpu().detach().numpy()
        test_y = batch_y.squeeze(-1).cpu().detach().numpy()
        test_pred += preds.tolist()
        test_truth += test_y.tolist()
best_test_loss=np.mean(test_loss)
best_test_acc = np.mean(test_acc)
best_test_acc_loss = np.mean(test_acc_loss)
print("Finished!\n----------------------------------------------------------------")
total_time = time.time() - start
print("Total Time:{},Preparation data Time:{},Training data Time:{}".format(total_time, preparedata_time,
                                                                            total_time - preparedata_time))
print("Valid Loss:{:.4f} Acc Loss{:.4f} Acc:{:.2f}%".format(best_val_loss, best_val_acc_loss, best_val_acc * 100.0))
print("Test Loss:{:.4f} Acc Loss{:.4f} Acc:{:.2f}%".format(best_test_loss, best_test_acc_loss, best_test_acc * 100.0))
Old_Evaluate(test_pred,test_truth)
New_Evaluate(test_pred,test_truth)

writer.flush()
writer.close()