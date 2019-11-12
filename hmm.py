import numpy as np

class HMM(object):

    # 保存隐含状态数目，初始化初始状态概率，状态转移概率
    def __init__(self, initial_prob, transition_prob):
        self.n_hidden = initial_prob.shape[0]
        self.initial_prob = initial_prob
        self.transition_prob = transition_prob

    # 似然和M部都与发射概率B有关, 而B可以连续可以离散, 故这里暂不定义
    def likelihood(self, Q):
        raise NotImplementedError

    def maximize(self, Qs, epsilon, gamma):
        raise NotImplementedError

    # 传入多组序列进行训练
    def fit(self, Qs, iter_max = 2):
        
        params = np.hstack((self.initial_prob.ravel(), self.transition_prob.ravel()))  # 将参数平铺开, 便于下面计算参数是否以一定精度不变
        
        # 其实这里应该也计算发射概率b的变化? d高斯时计算均值和协方差的变化

        for _ in range(iter_max):             # 约定_未不打算使用的变量: 防止出现未使用变量的警告 
            epsilon, gamma = self.expect(Qs)  # E部:计算似然函数Q的系数: 根据旧参数计算后验概率
            self.maximize(Qs, epsilon, gamma) # M部:计算似然函数Q的极值点, 得到新的参数, 并更新到类里面

            params_new = np.hstack((self.initial_prob.ravel(), self.transition_prob.ravel()))
            if np.allclose(params, params_new): # 逐元素 判断参数是否收敛, array中所有参数值收敛时返回True
                break
            else:
                params = params_new

        return 

    '''
    2
    E部:计算似然函数Q的系数: 根据旧参数计算后验概率
        epsilon[k,t,i,j] = P(zt = i, zt+1 = j | Qk), 第k组观测值的条件下, t时刻 zt = i, zt+1 = j (状态对i,j)出现的概率
        gamma[k,t,i] = P(zt = i| Qk), 第k组观测值的条件下, t时刻 zt = i (状态i)出现的概率 
        使用前向变量alpha, 后向变量belta, 用DP简化计算
        alpha[t,i] = P(o1,...,ot, zt = i)
        belta[t,i] = P(ot+1,...,oT | zt=i)
        性质:
        1.  P(o1,...,oT) 
            = sum_i alpha(T, i)
        2.  P(o1,...,oT, zt=i)  =   P(o1,...,ot, zt=i) * P(ot+1,...,oT | zt=i, o1,...,ot) = P(o1,...,ot, zt=i) * P(ot+1,...,oT | zt=i)
            = alpha(t,i) * belta(t,i)
        3.  P(o1,...,oT) 
            = sum_i alpha(t,i) * belta(t,i)
        epsilon, gamma可用alpha, belta表示
        epsilon[k,t,i,j] = P(zt = i, zt+1 = j | Qk) = P(zt = i, zt+1 = j, Qk) / P(Qk) 
                         = alpha(t,i) * a(i,j) * b(t+1,j) * belta(t+1,j) / P(Qk)
                         = alpha(t,i) * a(i,j) * b(t+1,j) * belta(t+1,j) / sum_i alpha(Tk, i)
        gamma[k,t,i] = P(zt = i| Qk) 
                     = sum_j epsilon[k,t,i,j]
    M部在子类中根据不同的发射概率模型实现 
    '''
    def expect(self, Qs):
        alphas = self.forward(Qs)
        beltas = self.backward(Qs)
        epsilons = list()
        gammas = list()
        for k, Q in enumerate(Qs): # 对Qs中每个Q计算一次epsilon, gamma, 加入列表epsilons, gammas
            T = Q.shape[0]
            alpha = alphas[k]
            belta = beltas[k]
            likelihood = self.likelihood(Q)

            epsilon = np.zeros((T, self.n_hidden, self.n_hidden))      
            for t in range(T-1):
                for i in range(self.n_hidden):
                    for j in range(self.n_hidden):
                        epsilon[t][i][j] = alpha[t][i] * self.transition_prob[i][j] * likelihood[t+1,j] * belta[t+1,j]
            Q_prob = alpha[T-1].sum()
            epsilon /= Q_prob
            epsilons.append(epsilon)

            gamma = np.zeros((T, self.n_hidden))
            for t in range(T):
                for i in range(self.n_hidden):
                    gamma[t][i] = alpha[t,i]*belta[t,i] #epsilon[t][i].sum()
            gamma /= Q_prob
            gammas.append(gamma)

        return epsilons, gammas

    """
    前向算法
    input : Qs 同一HMM模型的多个观测序列
    output: alpha[k, t, i] 第k个序列alpha(t,i)的值
            alpha[k].shape == (Tk, n_hidden)
    DP algorithm:
      define: alpha(t,i) = P(o1,...,ot,zt=i)
      init  : alpha(1,i) = P(o1,z1=i) 
                          = P(z1=i)P(o1|z1=i)
              alpha(1,) = init_prob * b_1
      more  : alpha(t+1,i) = P(o1,...,ot+1,zt+1=i) 
                          = sum_j P(o1,...,ot+1,zt=j, zt+1=i) 
                          = sum_j P(o1,...,ot,zt=j) P(ot+1,zt+1=i|zt=j,o1,...,ot)
                          = sum_j alpha(t, j) P(ot+1,zt+1=i|zt=j)
                          = sum_j alpha(t, j) P(zt+1=i|zt=j)P(ot+1|zt+1=i,zt=j)
                          = ( sum_j alpha(t, j) a[j,i] ) b[t+1,i]
              alpha(t+1,i) = alpha(t,) X a[,i] * b[t+1,i]
              alpha(t+1,)  = alpha(t,) X a[,]  * b[t+1,]
    """
    def forward(self, Qs):
        
        alphas = list()
        for Q in Qs:
            T = Q.shape[0]   # 注意: shape, 不需要加括号
            likelihood = self.likelihood(Q)  # 求出各时刻各状态发射观察值的概率
            
            alpha = list()
            alpha_1 = self.initial_prob * likelihood[0] # 初始时刻的alpha
            alpha.append(alpha_1)
            for t in range(1, T):
                alpha_t = np.matmul(alpha[-1], self.transition_prob) * likelihood[t]
                alpha.append(alpha_t)
            alpha = np.asarray(alpha)
            alphas.append(alpha)
            
            for t in range(1,T):
                for i in range(self.n_hidden):
                    #alpha(t+1,i) = ( sum_j alpha(t, j) a[j,i] ) b[t+1,i]
                    x = 0
                    for j in range(self.n_hidden):
                        x += alpha[t-1, j] * self.transition_prob[j][i] * likelihood[t][i]
                    if abs(x-alpha[t, i]) > 1e-3:
                        return False

        return alphas

    '''
    后向算法
    input : Qs 同一HMM模型的多个观测序列
    output: belta[k, t, i] belta(t,i)的值, t时刻状态i生成该时刻观测值的概率
            belta[k].shape == (Tk, n_hidden)
    DP algorithm:
      define: belta(t, i) = P(ot+1,...,oT|zt=i)
      init  : belta(T, i) = 1  :make sure P(o1,...,oT,zT=i) = alpha(T,i)*belta(T,i)
      more  : belta(t, i) = P(ot+1,...,oT|zt=i)
                          = sum_j P(zt+1=j,ot+1,ot+2,...,oT|zt=i)
                          = sum_j P(zt+1=j,ot+1|zt=i) P(ot+2,...,oT|zt+1=j,ot+1,zt=i)
                          = sum_j P(zt+1=j|zt=i) P(ot+1|zt+1=j,zt=i) P(ot+2,...,oT|zt+1=j)
                          = sum_j a(i,j) (b(t+1,j) belta(t+1,j))
                          = a(i,) X ( b(t+1,) * belta(t+1,) ) = a number
              belta(t, )  = [a(,)  X ( b(t+1,) * belta(t+1,) )] = an array, so don't need traverse
    '''
    def backward(self, Qs):
        beltas = list()
        for Q in Qs:
            T = Q.shape[0]
            likelihood = self.likelihood(Q)  # 求出各时刻各状态发射观察值的概率
            
            belta = list()
            belta_T = np.ones(self.n_hidden) # 初始时刻的belta
            belta.insert(0, belta_T)
            for t in range(T-2, -1, -1): # 这里及上文 t从T开始直到1, 故t时刻的似然存储在likelihood[t-1]内
                belta_t = np.matmul(self.transition_prob, likelihood[t+1] * belta[0])
                belta.insert(0, belta_t) # python list method: list.insert(index, obj) 在指定位置插入元素
            belta = np.asarray(belta)
            
            for t in range(1,T-1):
                for i in range(self.n_hidden):
                    # belta(t, i) = sum_j a(i,j) (b(t+1,j) belta(t+1,j))
                    x = 0
                    for j in range(self.n_hidden):
                        x += self.transition_prob[i][j] * likelihood[t+1][j] * belta[t+1][j]
                    if np.abs(belta[t,i] - x) > 1e-3:
                        return False   
            beltas.append(belta)

        return beltas

