import networkx as nx
import time
from functools import cmp_to_key
from preprocess import config
import sys
import random
import math

class IndexDict:
    def __init__(self, original_ids):
        self.original_to_new = {}
        self.new_to_original = []
        cnt = 0
        for i in original_ids:
            new = self.original_to_new.get(i, cnt)
            if new == cnt:
                self.original_to_new[i] = cnt
                cnt += 1
                self.new_to_original.append(i)

    def new(self, original):
        if type(original) is int:
            return self.original_to_new[original]
        else:
            if type(original[0]) is int:
                return [self.original_to_new[i] for i in original]
            else:
                return [[self.original_to_new[i] for i in l] for l in original]

    def original(self, new):
        if type(new) is int:
            return self.new_to_original[new]
        else:
            if type(new[0]) is int:
                return [self.new_to_original[i] for i in new]
            else:
                return [[self.new_to_original[i] for i in l] for l in new]

    def length(self):
        return len(self.new_to_original)


def gen_cascades_obser(observation_time, pre_times, filename,seed = 0):
    # a list to save the cascades
    cascades_type = dict()  # 0 for train, 1 for val, 2 for test
    cascades_time_dict = dict()
    cascades_total = 0
    cascades_valid_total = 0

    # Important node: for weibo dataset, if you want to compare CasFlow with baselines such as DeepHawkes and CasCN,
    # make sure the ob_time is set consistently.
    if observation_time in [3600, 3600 * 2, 3600 * 3]:  # end_hour is set to 19 in DeepHawkes and CasCN, but it should be 18
        end_hour = 19
    else:
        end_hour = 18

    with open(filename) as file:
        for line in file:
            cascades_total += 1
            parts = line.split('\t')
            cascade_id = parts[0]
            # filter cascades by their publish date/time
            if filename.__contains__("weibo"):
                # timezone invariant
                hour = int(time.strftime('%H', time.gmtime(float(parts[2])))) + 8
                if hour < 8 or hour >= end_hour:
                    continue
            elif filename.__contains__("twitter"):
                month = int(time.strftime('%m', time.localtime(float(parts[2]))))
                day = int(time.strftime('%d', time.localtime(float(parts[2]))))
                if month == 4 and day > 10:
                    continue
            elif filename.__contains__("aps"):
                publish_time = parts[2]
                if publish_time > '1997':
                    continue
            else:
                pass

            paths = parts[4].strip().split(' ')
            # number of observed popularity
            p_o = 0
            for p in paths:
                # observed adoption/participant
                nodes = p.split(':')[0].split('/')
                time_now = int(p.split(':')[1])
                if time_now < observation_time:
                    p_o += 1

            # filter cascades which observed popularity less than 10
            if p_o < 10:
                continue
            # for each cascade, save its publish time into a dict
            if filename.__contains__("aps"):
                cascades_time_dict[cascade_id] = int(0)
            else:
                cascades_time_dict[cascade_id] = int(parts[2])

            # write data into the targeted file, if they are not excluded
            cascades_valid_total += 1

    # open three files to save train, val, and test set, respectively

    def shuffle_cascades():
        # shuffle all cascades
        shuffle_time = list(cascades_time_dict.keys())
        random.seed(seed)
        random.shuffle(shuffle_time)

        count = 0
        a=0
        b=0
        c=0
        # split dataset
        for key in shuffle_time:
            if count < cascades_valid_total * .7:
                cascades_type[key] = 1  # training set, 70%
                a+=1
            elif count < cascades_valid_total * .85:
                cascades_type[key] = 2  # validation set, 15%
                b += 1
            else:
                cascades_type[key] = 3  # test set, 15%
                c += 1
            count += 1
        print("Number of valid train cascades: {}".format(a))
        print("Number of valid   val cascades: {}".format(b))
        print("Number of valid  test cascades: {}".format(c))
    shuffle_cascades()

    return cascades_valid_total, cascades_type


def discard_cascade(observation_time, pre_times, filename,cascades_type):
    discard_cascade_id = dict()
    count=0
    with open(filename) as f:
        for line in f:
            parts = line.split("\t")
            cascadeID = parts[0]
            if cascadeID not in cascades_type:
                continue
            path = parts[4].strip().rstrip('\n').split(" ")
            observation_path = []
            edges = set()
            for p in path:
                nodes = p.split(":")[0].split("/")
                if len(p.split(':')) < 2:
                    continue
                time_now = int(p.split(":")[1])
                if time_now < observation_time:
                    observation_path.append((nodes, time_now))
            observation_path.sort(key=lambda tup: tup[1])
            for i in range(min(config.max_seq,len(observation_path))):
                nodes = observation_path[i][0]
                if len(nodes)>1:
                    edges.add(nodes[-2] + ":" + nodes[-1] + ":1")
            nx_Cass = nx.DiGraph()
            for i in edges:
                part = i.split(":")
                source = part[0]
                target = part[1]
                weight = part[2]
                nx_Cass.add_edge(source, target, weight=weight)
            try:
                L = directed_laplacian_matrix(nx_Cass)
            except:
                discard_cascade_id[cascadeID] = 1
                count+=1
                s = sys.exc_info()
            else:
                num = nx_Cass.number_of_nodes()
    #             if num > config.max_seq:
    #                 discard_cascade_id[cascadeID] = 1
    #                 count += 1
    print("directed_laplacian_matrix delete count:",count)
    return discard_cascade_id


def directed_laplacian_matrix(G, nodelist=None, weight='weight',
                              walk_type=None, alpha=0.95):
    import scipy as sp
    from scipy.sparse import identity, spdiags, linalg
    if walk_type is None:
        if nx.is_strongly_connected(G):
            if nx.is_aperiodic(G):
                walk_type = "random"
            else:
                walk_type = "lazy"
        else:
            walk_type = "pagerank"

    M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight,
                                  dtype=float)
    n, m = M.shape
    if walk_type in ["random", "lazy"]:
        DI = spdiags(1.0 / sp.array(M.sum(axis=1).flat), [0], n, n)
        if walk_type == "random":
            P = DI * M
        else:
            I = identity(n)
            P = (I + DI * M) / 2.0

    elif walk_type == "pagerank":
        if not (0 < alpha < 1):
            raise nx.NetworkXError('alpha must be between 0 and 1')
        M = M.todense()
        dangling = sp.where(M.sum(axis=1) == 0)
        for d in dangling[0]:
            M[d] = 1.0 / n
        M = M / M.sum(axis=1)
        P = alpha * M + (1 - alpha) / n
    else:
        raise nx.NetworkXError("walk_type must be random, lazy, or pagerank")

    evals, evecs = linalg.eigs(P.T, k=1, tol=1E-2)
    v = evecs.flatten().real
    p = v / v.sum()
    sqrtp = sp.sqrt(p)
    Q = spdiags(sqrtp, [0], n, n) * P * spdiags(1.0 / sqrtp, [0], n, n)
    I = sp.identity(len(G))
    return I - (Q + Q.T) / 2.0

def gen_cascade(observation_time, pre_times, filename, filename_ctrain, filename_cval,
                filename_ctest, filename_strain, filename_sval, filename_stest, cascades_type, discard_cascade_id,seed=0):
    file = open(filename, "r")
    file_ctrain = open(filename_ctrain, "w")
    file_cval = open(filename_cval, "w")
    file_ctest = open(filename_ctest, "w")
    file_strain = open(filename_strain, "w")
    file_sval = open(filename_sval, "w")
    file_stest = open(filename_stest, "w")
    filtered_data=list()
    cascades_total=0
    for line in file:
        cascades_total+=1
        parts = line.split("\t")
        cascadeID = parts[0]
        if cascadeID in discard_cascade_id and discard_cascade_id[cascadeID]==1:
            continue
        if cascadeID not in cascades_type:
            continue
        path = parts[4].strip().rstrip('\n').split(" ")
        observation_path = []
        edges = []
        labels=0
        for p in path:
            nodes = p.split(":")[0].split("/")
            if len(p.split(':')) < 2:
                continue
            time_now = int(p.split(":")[1])
            if time_now < observation_time:
                observation_path.append((nodes, time_now))
            for i in range(len(pre_times)):
                if time_now < pre_times[i]:
                    labels += 1
        observation_path.sort(key=lambda tup: tup[1])
        sz_size=len(observation_path)
        labels = str(labels - sz_size)
        for i in range(min(config.max_seq,len(observation_path))):
            nodes = observation_path[i][0]
            time_now=observation_path[i][1]
            if len(nodes) <2:
                continue
            if (nodes[-2] + ":" + nodes[-1] + ":" + str(time_now)) in edges:
                continue
            else:
                edges.append(nodes[-2] + ":" + nodes[-1] + ":" + str(time_now))
        o_path = list()
        for i in range(min(config.max_seq,len(observation_path))):
            nodes = observation_path[i][0]
            t = observation_path[i][1]
            o_path.append(','.join(nodes) + ':' + str(t))
        observation_path=o_path
        c_line = cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(sz_size) + "\t" + " ".join(edges) + "\t" +labels+ "\n"
        s_line= cascadeID + "\t" + "\t".join(observation_path) + "\n"
        filtered_data.append([c_line,s_line])
    filtered_data_train = list()
    filtered_data_val = list()
    filtered_data_test = list()
    for line in filtered_data:
        cascade_id = line[0].split('\t')[0]
        if cascade_id in cascades_type and cascades_type[cascade_id] == 1 :
            filtered_data_train.append(line)
        elif cascade_id in cascades_type and cascades_type[cascade_id] == 2 :
            filtered_data_val.append(line)
        elif cascade_id in cascades_type and cascades_type[cascade_id] == 3:
            filtered_data_test.append(line)
        else:
            print('What happened?')
    print("Number of valid cascades: {}/{}".format(len(filtered_data),cascades_total))
    print("Number of valid train cascades: {}".format(len(filtered_data_train)))
    print("Number of valid   val cascades: {}".format(len(filtered_data_val)))
    print("Number of valid  test cascades: {}".format(len(filtered_data_test)))
    random.seed(seed)
    random.shuffle(filtered_data_train)
    def file_write(c_file,s_file,line):
        c_file.write(line[0])
        s_file.write(line[1])
    for line in filtered_data_train + filtered_data_val + filtered_data_test:
        cascade_id = line[0].split('\t')[0]
        if cascade_id in cascades_type and cascades_type[cascade_id] == 1:
            file_write(file_ctrain,file_strain,line)
        elif cascade_id in cascades_type and cascades_type[cascade_id] == 2:
            file_write(file_cval,file_sval,line)
        elif cascade_id in cascades_type and cascades_type[cascade_id] == 3:
            file_write(file_ctest,file_stest,line)
    file.close()
    file_ctrain.close()
    file_cval.close()
    file_ctest.close()
    file_strain.close()
    file_sval.close()
    file_stest.close()

def get_original_ids(graphs):
    original_ids = set()
    for graph in graphs.keys():
        for walk in graphs[graph]:
            for i in walk[0]:
                original_ids.add(i)
    print("length of original isd:", len(original_ids))
    return original_ids


def sequence2list(flename):
    graphs = {}
    with open(flename, 'r') as f:
        for line in f:
            walks = line.strip().split('\t')
            graphs[walks[0]] = []  # walk[0] = cascadeID
            for i in range(1, len(walks)):
                s = walks[i].split(":")[0]  # node
                t = walks[i].split(":")[1]  # time
                graphs[walks[0]].append([[int(xx) for xx in s.split(",")], int(t)])
    return graphs


if __name__ == "__main__":
    observation_time = config.observation
    pre_times = config.pre_times

    cascades_total, cascades_type = gen_cascades_obser(observation_time, pre_times, config.cascades)

    discard_cascade_id = discard_cascade(observation_time, pre_times, config.cascades,cascades_type)

    print("generate cascade new!!!")
    gen_cascade(observation_time, pre_times, config.cascades, config.cascade_train,
                config.cascade_val, config.cascade_test,
                config.shortestpath_train, config.shortestpath_val,
                config.shortestpath_test,
                cascades_type, discard_cascade_id)






