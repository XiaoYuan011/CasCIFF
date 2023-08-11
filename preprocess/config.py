DATA_PATHA='./../dataset/CasCIFF/weibo'
time_interval="/0.5h"

# DATA_PATHA='./../dataset/CasCIFF/twitter'
# time_interval="/1day"

# DATA_PATHA='./../dataset/CasCIFF/aps'
# time_interval="/3year"

cascades  = DATA_PATHA+"/dataset.txt"
#node increment for each snapshot
increase_node=1

data_type="CasCN_adj_step{}".format(increase_node)

DATA_PATHA=DATA_PATHA+time_interval
cascade_train = DATA_PATHA+"/cascade_train.txt"
cascade_val = DATA_PATHA+"/cascade_val.txt"
cascade_test = DATA_PATHA+"/cascade_test.txt"
shortestpath_train = DATA_PATHA+"/shortestpath_train.txt"
shortestpath_val = DATA_PATHA+"/shortestpath_val.txt"
shortestpath_test = DATA_PATHA+"/shortestpath_test.txt"
train_pkl = DATA_PATHA+"/data_train_{}.pkl".format(data_type)
val_pkl = DATA_PATHA+"/data_val_{}.pkl".format(data_type)
test_pkl = DATA_PATHA+"/data_test_{}.pkl".format(data_type)

max_seq=100
pre_times = [0]
observation = 0
# observation and prediction time settings:
# for twitter dataset, we use 3600*24*1 (86400, 1 day) or 3600*24*2 (172800, 2 days) as observation time
#                      we use 3600*24*32 (2764800, 32 days) as prediction time
# for weibo   dataset, we use 1800 (0.5 hour) or 3600 (1 hour) as observation time
#                      we use 3600*24 (86400, 1 day) as prediction time
# for aps     dataset, we use 365*3 (1095, 3 years) or 365*5+1 (1826, 5 years) as observation time
#                      we use 365*20+5 (7305, 20 years) as prediction time
if time_interval.__contains__("1day"):
    observation = 3600 * 24 * 1
    pre_times = [24 * 3600 * 32]
    n_time_interval = (observation) // (3 * 60 * 60)
elif time_interval.__contains__("2day"):
    observation = 3600 * 24 * 2
    pre_times = [24 * 3600 * 32]
    n_time_interval = (observation) // (3 * 60 * 60)
elif time_interval.__contains__("0.5h"):
    pre_times = [24 * 3600]
    observation =1800*1
    n_time_interval = 6  # 5m
elif time_interval.__contains__("1h"):
    pre_times = [24 * 3600]
    observation =1800*2
    n_time_interval = 6  # 10m
elif time_interval.__contains__("3year"):
    observation =365*3
    pre_times = [365*20+5]
    n_time_interval = (observation)//(3*30)  #3month
elif time_interval.__contains__("5year"):
    observation = 365*5+1
    pre_times = [365 * 20 + 5]
    n_time_interval = (observation)//(3*30)  #3month
