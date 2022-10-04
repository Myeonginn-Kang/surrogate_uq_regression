import numpy as np


def data_summary(dname):
 
    dfile = 'data/%s.csv'%dname    
    data = np.genfromtxt(dfile, delimiter=',')
    dim_x = data.shape[1]-1
    
    print(dname)
    print('-- dataset size: %d, n. features: %d'%(len(data), dim_x))


for dname in ['bikesharing','compactiv','cpusmall','ctscan','indoorloc','mv','pole','puma32h','telemonitoring']:
    data_summary(dname)