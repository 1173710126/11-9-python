import struct
import numpy as np
import os
from GaussianHMM import GaussianHMM
# input: 特征文件夹的路径
# return: datas, 各类别的音频特征组成的列表
#   其中datas['1'] = a list :属于类别1的特征列表, 列表中每个元素都是一个特征序列(T*n_dim的numpy矩阵)
def get_mfc_data(path):
    files = os.listdir(path)
    datas = dict()
    for file_name in files: # 读取每个mfc文件到矩阵data中
        data = list()
        with open(path+file_name, 'rb') as f:
            nframes = struct.unpack('>i',f.read(4))[0] # 帧数 
            _ = struct.unpack('>i',f.read(4))[0]   # 帧移，100ns为单位，100000指10ms，
            nbytes = struct.unpack('>h',f.read(2))[0]  # 每帧特征值的字节长度
            ndim = nbytes / 4                            # 每帧的特征的维度（一维为一个int）
            _ = struct.unpack('>h',f.read(2))[0] # [没用] 用户序号
            while True:
                data_byte = f.read(4)
                if len(data_byte) < 4: 
                    break
                data.append(struct.unpack('>f', data_byte)[0])   
        data = np.array(data)
        data.shape = nframes, int(ndim)
        category = file_name[0]
        if category in datas:
            datas[category].append(data)
        else:
            datas[category] = list()
            datas[category].append(data)
    return datas

if __name__ == "__main__":
    datas = get_mfc_data('C:/Users/18341/Desktop/book/听觉/实验3-语音识别/语料/features/')
    '''
    for key in datas:
        print(len(datas[key]))
        print(len(datas[key][0]))
    '''
    hmms = dict()
        
    '''
    for category in datas:
        Qs = datas[category]
        n_hidden = 6
        #initial_prob = np.random.randn(n_hidden)
        #transition_prob = np.random.randn(n_hidden, n_hidden)
        initial_prob = np.ones((n_hidden))
        initial_prob /= n_hidden
        transition_prob = np.ones((n_hidden, n_hidden))
        transition_prob /= n_hidden
        
        n_dim = len(Qs[0][0])
        means = np.random.randn(n_hidden, n_dim)  
        covs = np.random.randn(n_hidden, n_dim, n_dim)
        for i in range(n_hidden):
            covs[i] = np.eye(n_dim, n_dim)

        hmm = GaussianHMM(initial_prob, transition_prob, means, covs)
        hmm.viterbi_init(Qs, iter_max=5)
        print('success viterbi_init')
        hmm.fit(Qs[:-3], iter_max = 10)
        print('success fit')
        hmms[category] = hmm
    '''  
    for category in datas:
        Qs = datas[category]
        n_hidden = 3
        #initial_prob = np.random.randn(n_hidden)
        #transition_prob = np.random.randn(n_hidden, n_hidden)
        initial_prob = np.ones((n_hidden))
        initial_prob /= n_hidden
        transition_prob = np.ones((n_hidden, n_hidden))
        transition_prob /= n_hidden
        
        n_dim = len(Qs[0][0])
        means = np.random.randn(n_hidden, n_dim)  
        covs = np.random.randn(n_hidden, n_dim, n_dim)
        for i in range(n_hidden):
            covs[i] = np.eye(n_dim, n_dim)

        hmm = GaussianHMM(initial_prob, transition_prob, means, covs)
        hmm.viterbi_init(Qs, iter_max=5)
        hmms[category] = hmm

    evaluate_num = 5
    for evaluate_cnt in range(evaluate_num):
        print(evaluate_cnt, 'start fit')
        for category in hmms:
            hmm = hmms[category]
            Qs = datas[category]
            hmm.fit(Qs[:-3], iter_max = 20)
            hmms[category] = hmm
        
        # test
        correct_num = 0
        for category in datas:
            for test_sample in datas[category][-3:]:
                print('real_category:', category)
                max_like = -1 * np.inf
                predict = -1
                for predict_category in hmms:
                    hmm = hmms[predict_category]
                    like = hmm.generate_prob(test_sample)
                    print('category', predict_category, '. like:', like)
                    if like > max_like:
                        max_like = like
                        predict = predict_category
                        #print('predict_category', predict_category)
                if predict == category:
                    correct_num += 1
                print('predict_category:',predict)
        print(correct_num / (3*5))
    