import tushare as ts
import pandas as pd
import numpy as np

code_list = ts.get_today_all()['code'].to_numpy()
data_list = []
code_list2 = []
for code in code_list:
    tmp = ts.get_hist_data(code)
    if tmp is not None:
        data_list.append(tmp.to_numpy())
        code_list2.append(code)
np.save('data.npy', np.array(data_list))
np.save('code.npy', np.array(code_list2))



#path = '/mnt/ufs/team39/'
path = ''
data = np.load(path+'data.npy')
code = np.load(path+'code.npy')
len_fix = 610
l = 0
data_zjz = []
code_zjz = []
for i in range(data.shape[0]):
    if data[i].shape[0] == len_fix:
        data_zjz.append(data[i])
        code_zjz.append(code[i])
        l += 1
        if l == 300:
            break
np.save('data_zjz.npy', np.array(data_zjz))
np.save('code_zjz.npy', np.array(code_zjz))




#data_zjz = np.load('data_zjz.npy')
#code_zjz = np.load('code_zjz.npy')
#print(data_zjz.shape)
#print(code_zjz.shape)








#tmp = ts.get_hist_data('688388')
#print(tmp.index)
#print(tmp.columns)

#tmp = ts.get_today_all()
#tmp = np.array(tmp['code'].to_numpy())
#np.save('code.npy', tmp)
#print(tmp.dtype)
#print(tmp.shape)

'''
code_list = list(np.load('/mnt/ufs/team39/code.npy'))
data_list = []
new_code_list = []
for code in code_list:
    tmp = ts.get_hist_data(code)
    if tmp is not None:
        data_list.append(tmp.to_numpy())
        new_code_list.append(code)
data = np.array(data_list)
code = np.array(new_code_list)
np.save('data.npy', data)
np.save('code.npy', code)
'''

'''
path = '/mnt/ufs/team39/'

data = np.load(path+'data.npy')
code = np.load(paht+'code.npy')
len_fix = 610
new_data = []
new_code = []
for i in range(data.shape[0]):
    if data[i].shape[0] == len_fix:
        new_data.append(data[i])
        new_code.append(code[i])
np.save('data_zjz.npy', new_data)
np.save('code_zjz.npy', new_code)
'''


