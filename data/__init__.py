# encoding=utf-8

# create_dataset(data_dir, train),从.H5文件读取数据
def create_dataset(data_dir, train):
    dataset = None
    if data_dir == 'data/DE.h5' or data_dir == 'data/FE.h5':
        from .custom_dataset import CWRUdata
        dataset = CWRUdata(data_dir, train)
    else:
        raise ValueError("Dataset [%s] not recognized." % data_dir)
    print("dataset [%s] was created" % data_dir)
    return dataset