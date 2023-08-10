import pickle
import time

start = time.time()
max_nodes = 50
node_level = {}
data_type = 'weibo'
interval = '/0.5h'
isordered = '6nei_sample{}'.format(max_nodes)
Data_Base = './../../../dataset/GTSSpaceNet/' + data_type
Data_path = './../../../dataset/GTSSpaceNet/{0}/global/global_emb_'.format(data_type) + isordered + '.txt'
with open(Data_path, 'r') as f:
    for line in f:
        parts = line.rstrip('\n').split('\t')
        node_level[parts[0]] = int(parts[-1])
a = sorted(node_level.items(), key=lambda x: x[1], reverse=True)
'''
Experiment:
    Two categories: important people/non-important people
    Initial classification index: Top 30%
'''
save_file = Data_Base + interval
node_class = {}
count = 0
top = 30
for node in a:
    if count < len(a) // top:
        node_class[node[0]] = 1
    else:
        node_class[node[0]] = 0
    count += 1
# Node embedding
node_emb = {}
print('start process generate data...')
with open(Data_path, 'r') as f:
    for line in f:
        parts = line.rstrip('\n').split('\t')
        node_id = parts[0]
        node_features = []
        negF_degrees = []
        negS_degrees = []
        negT_degrees = []
        negV_degrees = []
        negVV_degrees = []
        negSS_degrees = []
        if isordered.__contains__("0nei"):
            if not node_emb.__contains__(node_id):
                drgree = parts[-1]
                if degree != '':
                    node_emb[node_id] = int(degree)
            continue
        length = 0
        for degree in parts[1].split(' '):
            if degree == '':
                continue
            negF_degrees.append(int(degree))
            length += 1
        while length < max_nodes:
            negF_degrees.append(0)
            length += 1
        node_features.append(negF_degrees)
        if isordered.__contains__("1nei"):
            if not node_emb.__contains__(node_id):
                node_emb[node_id] = node_features
            continue
        length = 0
        for degree in parts[2].split(' '):
            if degree == '':
                continue
            negS_degrees.append(int(degree))
            length += 1
        while length < max_nodes:
            negS_degrees.append(0)
            length += 1
        node_features.append(negS_degrees)
        if isordered.__contains__("2nei"):
            if not node_emb.__contains__(node_id):
                node_emb[node_id] = node_features
            continue
        length = 0
        for degree in parts[3].split(' '):
            if degree == '':
                continue
            negT_degrees.append(int(degree))
            length += 1
        while length < max_nodes:
            negT_degrees.append(0)
            length += 1
        node_features.append(negT_degrees)
        if isordered.__contains__("3nei"):
            if not node_emb.__contains__(node_id):
                node_emb[node_id] = node_features
            continue
        length = 0
        for degree in parts[4].split(' '):
            if degree == '':
                continue
            negV_degrees.append(int(degree))
            length += 1
        while length < max_nodes:
            negV_degrees.append(0)
            length += 1
        node_features.append(negV_degrees)
        if isordered.__contains__("4nei"):
            if not node_emb.__contains__(node_id):
                node_emb[node_id] = node_features
            continue
        length = 0
        for degree in parts[5].split(' '):
            if degree == '':
                continue
            negVV_degrees.append(int(degree))
            length += 1
        while length < max_nodes:
            negVV_degrees.append(0)
            length += 1
        node_features.append(negVV_degrees)
        if isordered.__contains__("5nei"):
            if not node_emb.__contains__(node_id):
                node_emb[node_id] = node_features
            continue
        length = 0
        for degree in parts[6].split(' '):
            if degree == '':
                continue
            negSS_degrees.append(int(degree))
            length += 1
        while length < max_nodes:
            negSS_degrees.append(0)
            length += 1
        node_features.append(negSS_degrees)
        if not node_emb.__contains__(node_id):
            node_emb[node_id] = node_features

train_file = save_file + '/shortestpath_train.txt'
test_file = save_file + '/shortestpath_test.txt'
val_file = save_file + '/shortestpath_val.txt'
print('start process train data...')

start_time = time.time()
count = 0
f = open(train_file, 'r')
total_count = len(f.readlines())
f.close()
with open(train_file, 'r') as f:
    x_data_train = []
    y_data_train = []
    for line in f:
        nodes = []
        parts = line.rstrip('\n').split('\t')
        for path in parts[1:]:
            path = path.split(':')
            t = int(path[-1])
            node = path[0].split(',')[-1]
            nodes.append((node, t))
        nodes.sort(key=lambda tup: tup[1])
        x_data = []
        y_data = []
        for i in range(min(len(nodes), 100)):
            node = nodes[i][0]
            x_data.append(node_emb[node])
            y_data.append(node_class[node])
        x_data_train.append(x_data)
        y_data_train.append(y_data)
        count += 1
        if count % 1000 == 0:
            print('has processed {}/{} time:{}s'.format(count, total_count, time.time() - start_time))
            start_time = time.time()
    pickle.dump((x_data_train, y_data_train),
                open(save_file + '/global_step_train_' + isordered + '_top_' + str(top) + '.pkl', 'wb'))

print('start process val data...')
start_time = time.time()
count = 0
f = open(val_file, 'r')
total_count = len(f.readlines())
f.close()
with open(val_file, 'r') as f:
    x_data_val = []
    y_data_val = []
    for line in f:
        nodes = []
        parts = line.rstrip('\n').split('\t')
        for path in parts[1:]:
            path = path.split(':')
            t = int(path[-1])
            node = path[0].split(',')[-1]
            nodes.append((node, t))
        nodes.sort(key=lambda tup: tup[1])
        x_data = []
        y_data = []
        for i in range(min(len(nodes), 100)):
            node = nodes[i][0]
            x_data.append(node_emb[node])
            y_data.append(node_class[node])
        x_data_val.append(x_data)
        y_data_val.append(y_data)
        count += 1
        if count % 1000 == 0:
            print('has processed {}/{} time:{}s'.format(count, total_count, time.time() - start_time))
            start_time = time.time()
    pickle.dump((x_data_val, y_data_val),
                open(save_file + '/global_step_val_' + isordered + '_top_' + str(top) + '.pkl', 'wb'))

print('start process test data...')
start_time = time.time()
count = 0
f = open(test_file, 'r')
total_count = len(f.readlines())
f.close()
with open(test_file, 'r') as f:
    x_data_test = []
    y_data_test = []
    for line in f:
        nodes = []
        parts = line.rstrip('\n').split('\t')
        for path in parts[1:]:
            path = path.split(':')
            t = int(path[-1])
            node = path[0].split(',')[-1]
            nodes.append((node, t))
        nodes.sort(key=lambda tup: tup[1])
        x_data = []
        y_data = []
        for i in range(min(len(nodes), 100)):
            node = nodes[i][0]
            x_data.append(node_emb[node])
            y_data.append(node_class[node])
        x_data_test.append(x_data)
        y_data_test.append(y_data)
        count += 1
        if count % 1000 == 0:
            print('has processed {}/{} time:{}s'.format(count, total_count, time.time() - start_time))
            start_time = time.time()
    pickle.dump((x_data_test, y_data_test),
                open(save_file + '/global_step_test_' + isordered + '_top_' + str(top) + '.pkl', 'wb'))

print('Finished!!!')
print('total time {}s'.format(time.time() - start))
