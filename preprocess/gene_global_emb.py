import networkx as nx
import pickle
import time
import random

max_nodes = 50
nei_num = "2nei"
data_type = 'aps'
random.seed(0)


def find123Nei(G, node):
    nei1_li = {}
    nei2_li = {}
    nei3_li = {}
    nei4_li = {}
    nei5_li = {}
    nei6_li = {}
    for FNs in random.sample(list(nx.neighbors(G, node)),
                             min(max_nodes, len(list(nx.neighbors(G, node))))):  # find 1_th neighbors
        nei1_li[FNs] = nx.degree(G, FNs)
    if nei2_li.__contains__(node):
        nei1_li.pop(node)
    if nei_num == "1nei":
        nei1_drgees = random.sample(list(nei1_li.values()), min(len(nei1_li), max_nodes))
        return nei1_drgees
    for n1 in nei1_li.keys():
        for SNs in random.sample(list(nx.neighbors(G, n1)),
                                 min(max_nodes // 2, len(list(nx.neighbors(G, n1))))):  # find 2_th neighbors
            nei2_li[SNs] = nx.degree(G, SNs)
    [nei2_li.pop(n) for n in list(nei1_li.keys()) + [node] if nei2_li.__contains__(n)]
    if nei_num == "2nei":
        nei1_drgees = random.sample(list(nei1_li.values()), min(len(nei1_li), max_nodes))
        nei2_drgees = random.sample(list(nei2_li.values()), min(len(nei2_li), max_nodes))
        return nei1_drgees, nei2_drgees
    elif nei_num == "3nei":
        for n2 in nei2_li.keys():
            for TNs in random.sample(list(nx.neighbors(G, n2)), min(max_nodes // 4, len(list(nx.neighbors(G, n2))))):
                nei3_li[TNs] = nx.degree(G, TNs)
        [nei3_li.pop(n) for n in list(nei1_li.keys()) + list(nei2_li.keys()) + [node] if nei3_li.__contains__(n)]
        nei1_drgees = random.sample(list(nei1_li.values()), min(len(nei1_li), max_nodes))
        nei2_drgees = random.sample(list(nei2_li.values()), min(len(nei2_li), max_nodes))
        nei3_drgees = random.sample(list(nei3_li.values()), min(len(nei3_li), max_nodes))
        return nei1_drgees, nei2_drgees, nei3_drgees
    elif nei_num == "4nei":
        for n2 in nei2_li.keys():
            for TNs in random.sample(list(nx.neighbors(G, n2)), min(max_nodes // 4, len(list(nx.neighbors(G, n2))))):
                nei3_li[TNs] = nx.degree(G, TNs)
        [nei3_li.pop(n) for n in list(nei1_li.keys()) + list(nei2_li.keys()) + [node] if nei3_li.__contains__(n)]
        for n3 in nei3_li.keys():
            for FNs in random.sample(list(nx.neighbors(G, n3)), min(max_nodes // 8, len(list(nx.neighbors(G, n3))))):
                nei4_li[FNs] = nx.degree(G, FNs)
        [nei4_li.pop(n) for n in list(nei1_li.keys()) + list(nei2_li.keys()) + list(nei3_li.keys()) + [node] if
         nei4_li.__contains__(n)]
        nei1_drgees = random.sample(list(nei1_li.values()), min(len(nei1_li), max_nodes))
        nei2_drgees = random.sample(list(nei2_li.values()), min(len(nei2_li), max_nodes))
        nei3_drgees = random.sample(list(nei3_li.values()), min(len(nei3_li), max_nodes))
        nei4_drgees = random.sample(list(nei4_li.values()), min(len(nei4_li), max_nodes))
        return nei1_drgees, nei2_drgees, nei3_drgees, nei4_drgees
    elif nei_num == '5nei':
        for n2 in nei2_li.keys():
            for TNs in random.sample(list(nx.neighbors(G, n2)), min(max_nodes // 4, len(list(nx.neighbors(G, n2))))):
                nei3_li[TNs] = nx.degree(G, TNs)
        [nei3_li.pop(n) for n in list(nei1_li.keys()) + list(nei2_li.keys()) + [node] if nei3_li.__contains__(n)]
        for n3 in nei3_li.keys():
            for FNs in random.sample(list(nx.neighbors(G, n3)), min(max_nodes // 8, len(list(nx.neighbors(G, n3))))):
                nei4_li[FNs] = nx.degree(G, FNs)
        [nei4_li.pop(n) for n in list(nei1_li.keys()) + list(nei2_li.keys()) + list(nei3_li.keys()) + [node] if
         nei4_li.__contains__(n)]
        for n4 in nei4_li.keys():
            for FFNs in random.sample(list(nx.neighbors(G, n4)), min(max_nodes // 16, len(list(nx.neighbors(G, n4))))):
                nei5_li[FFNs] = nx.degree(G, FFNs)
        [nei5_li.pop(n) for n in
         list(nei1_li.keys()) + list(nei2_li.keys()) + list(nei3_li.keys()) + list(nei4_li.keys()) + [node] if
         nei5_li.__contains__(n)]
        nei1_drgees = random.sample(list(nei1_li.values()), min(len(nei1_li), max_nodes))
        nei2_drgees = random.sample(list(nei2_li.values()), min(len(nei2_li), max_nodes))
        nei3_drgees = random.sample(list(nei3_li.values()), min(len(nei3_li), max_nodes))
        nei4_drgees = random.sample(list(nei4_li.values()), min(len(nei4_li), max_nodes))
        nei5_drgees = random.sample(list(nei5_li.values()), min(len(nei5_li), max_nodes))
        return nei1_drgees, nei2_drgees, nei3_drgees, nei4_drgees, nei5_drgees
    elif nei_num == '6nei':
        for n2 in nei2_li.keys():
            for TNs in random.sample(list(nx.neighbors(G, n2)), min(max_nodes // 4, len(list(nx.neighbors(G, n2))))):
                nei3_li[TNs] = nx.degree(G, TNs)
        [nei3_li.pop(n) for n in list(nei1_li.keys()) + list(nei2_li.keys()) + [node] if nei3_li.__contains__(n)]
        for n3 in nei3_li.keys():
            for FNs in random.sample(list(nx.neighbors(G, n3)), min(max_nodes // 8, len(list(nx.neighbors(G, n3))))):
                nei4_li[FNs] = nx.degree(G, FNs)
        [nei4_li.pop(n) for n in list(nei1_li.keys()) + list(nei2_li.keys()) + list(nei3_li.keys()) + [node] if
         nei4_li.__contains__(n)]
        for n4 in nei4_li.keys():
            for FFNs in random.sample(list(nx.neighbors(G, n4)), min(max_nodes // 16, len(list(nx.neighbors(G, n4))))):
                nei5_li[FFNs] = nx.degree(G, FFNs)
        [nei5_li.pop(n) for n in
         list(nei1_li.keys()) + list(nei2_li.keys()) + list(nei3_li.keys()) + list(nei4_li.keys()) + [node] if
         nei5_li.__contains__(n)]
        for n5 in nei5_li.keys():
            for SSNs in random.sample(list(nx.neighbors(G, n5)), min(max_nodes // 16, len(list(nx.neighbors(G, n5))))):
                nei6_li[SSNs] = nx.degree(G, SSNs)
        [nei6_li.pop(n) for n in
         list(nei1_li.keys()) + list(nei2_li.keys()) + list(nei3_li.keys()) + list(nei4_li.keys()) + list(
             nei5_li.keys()) + [node] if nei6_li.__contains__(n)]

        nei1_drgees = random.sample(list(nei1_li.values()), min(len(nei1_li), max_nodes))
        nei2_drgees = random.sample(list(nei2_li.values()), min(len(nei2_li), max_nodes))
        nei3_drgees = random.sample(list(nei3_li.values()), min(len(nei3_li), max_nodes))
        nei4_drgees = random.sample(list(nei4_li.values()), min(len(nei4_li), max_nodes))
        nei5_drgees = random.sample(list(nei5_li.values()), min(len(nei5_li), max_nodes))
        nei6_drgees = random.sample(list(nei6_li.values()), min(len(nei6_li), max_nodes))
        return nei1_drgees, nei2_drgees, nei3_drgees, nei4_drgees, nei5_drgees, nei6_drgees


gg_path = './../dataset/CasCIFF/{}/global/global_graph.pkl'.format(data_type)
global_emb = './../dataset/CasCIFF/{0}/global/global_emb_{1}_sample{2}.txt'.format(data_type, nei_num,
                                                                                             max_nodes)
with open(gg_path, 'rb') as f:
    gg = pickle.load(f)
nodes = list(nx.nodes(gg))
total_count = len(nodes)
count = 0
f = open(global_emb, 'w')
start_time = time.time()
for node in nodes:
    neighbors = find123Nei(gg, node)
    if nei_num == "1nei":
        line = node + '\t' + ' '.join([str(x) for x in sorted(neighbors, reverse=True)]) + '\t' + str(
            nx.degree(gg, node)) + '\n'
    elif nei_num == "2nei":
        line = node + '\t' + ' '.join([str(x) for x in sorted(neighbors[0], reverse=True)]) + '\t' + ' '.join(
            [str(x) for x in sorted(neighbors[1], reverse=True)]) + '\t' + str(nx.degree(gg, node)) + '\n'
    elif nei_num == "3nei":
        line = node + '\t' + ' '.join([str(x) for x in sorted(neighbors[0], reverse=True)]) + '\t' + ' '.join(
            [str(x) for x in sorted(neighbors[1], reverse=True)]) + '\t' + ' '.join(
            [str(x) for x in sorted(neighbors[2], reverse=True)]) + '\t' + str(nx.degree(gg, node)) + '\n'
    elif nei_num == "4nei":
        line = node + '\t' + ' '.join(
            [str(x) for x in sorted(neighbors[0], reverse=True)]) + '\t' + ' '.join(
            [str(x) for x in sorted(neighbors[1], reverse=True)]) + '\t' + ' '.join(
            [str(x) for x in sorted(neighbors[2], reverse=True)]) + '\t' + ' '.join(
            [str(x) for x in sorted(neighbors[3], reverse=True)]) + '\t' + str(nx.degree(gg, node)) + '\n'
    elif nei_num == "5nei":
        line = node + '\t' + ' '.join(
            [str(x) for x in sorted(neighbors[0], reverse=True)]) + '\t' + ' '.join(
            [str(x) for x in sorted(neighbors[1], reverse=True)]) + '\t' + ' '.join(
            [str(x) for x in sorted(neighbors[2], reverse=True)]) + '\t' + ' '.join(
            [str(x) for x in sorted(neighbors[3], reverse=True)]) + '\t' + ' '.join(
            [str(x) for x in sorted(neighbors[4], reverse=True)]) + '\t' + str(nx.degree(gg, node)) + '\n'
    elif nei_num == "6nei":
        line = node + '\t' + ' '.join(
            [str(x) for x in sorted(neighbors[0], reverse=True)]) + '\t' + ' '.join(
            [str(x) for x in sorted(neighbors[1], reverse=True)]) + '\t' + ' '.join(
            [str(x) for x in sorted(neighbors[2], reverse=True)]) + '\t' + ' '.join(
            [str(x) for x in sorted(neighbors[3], reverse=True)]) + '\t' + ' '.join(
            [str(x) for x in sorted(neighbors[4], reverse=True)]) + '\t' + ' '.join(
            [str(x) for x in sorted(neighbors[5], reverse=True)]) + '\t' + str(nx.degree(gg, node)) + '\n'
    f.write(line)
    count += 1
    if count % 10000 == 0:
        print('has processed {}/{} time:{:.2f}s'.format(count, total_count, time.time() - start_time))
        start_time = time.time()
f.close()
