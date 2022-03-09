import numpy as np
import pandas as pd
from greymodel import gm11
import random
import math


class AdaboostGM(object):
    """
    定义基学习器为GM(1,1)模型的Adaboost回归

    使用方法：
    1.实例化对象  model = AdaboostGM(data)
    2.进行训练    values = model.fit()
    3.进行预测    pred = model.predict()
    4.查看损失    model.MSE(values)

    参数解释：
    data:原始数据
    max_baseLearner:基学习器个数
    target_acc:目标损失率
    predict_step:预测长度
    lr:学习率，用于控制预测符号序列的正负
    """

    def __init__(self, data: pd.DataFrame, max_baseLearner: int = 10, target_acc: float = 0.1, predict_step: int = 1,
                 lr: float = 1.0, a = 1):
        self.data = data
        self.data_shape = self.data.shape[0]
        self.max_baseLearner = max_baseLearner
        self.__init_weight = np.ones((1, self.data_shape)) / self.data_shape
        self.__init_weight = self.__init_weight[0]
        self.target_acc = target_acc
        self.error = []
        self.__weight_coff = 1.1
        self.__weight_coff_lt = []
        self.predict_step = predict_step
        self.__coff_matrix = np.zeros((self.max_baseLearner + 1, 2))  # 系数矩阵，用来保存训练过程中模型的系数
        self.__predict_pm = np.zeros((self.max_baseLearner + 1, 2))
        self.predict_array = np.zeros((self.predict_step, 3))
        self.lr = lr
        # 保存最近的n个符号
        self.__pmpred_mat = np.zeros((self.max_baseLearner, self.predict_step))
        self.a = a

    # 利用误差更新权重
    def __update_weight(self, weights: np.array, errors: np.array) -> list:
        errors = abs(np.array(errors))
        # 获得最大误差
        max_error = max(errors)
        # 计算相对误差
        rel_errors = []
        for i in range(self.data_shape):
            rel_errors.append(errors[i] / max_error)
        # 计算回归误差率
        regress_rate = 0
        for i in range(self.data_shape):
            regress_rate += weights[i] * rel_errors[i]
        # 计算基学习器权重系数
        self.__weight_coff = regress_rate / (1 - regress_rate)
        self.__weight_coff_lt.append(self.__weight_coff)
        # 更新权重
        Z = 0
        for i in range(self.data_shape):
            Z += weights[i] * self.__weight_coff ** (1 - errors[i])
        new_weights = []
        for i in range(self.data_shape):
            new_weights.append((weights[i] * self.__weight_coff ** (1 - errors[i])) / Z)
        return new_weights

    # 使用新的权重更新背景值系数
    def __update_background_value(self, new_weights: list) -> list:
        new_background_value = []
        for i in range(1, self.data_shape):
            new_background_value.append(new_weights[i - 1] / (new_weights[i] + new_weights[i - 1] + 0.01))
        return new_background_value

    # 计算符号序列
    def __count_pm(self, residual: np.array) -> list:
        residual_lt = []
        for i in residual:
            if i >= 0:
                residual_lt.append(1)
            else:
                residual_lt.append(-1)
        return residual_lt

    # 定义符号序列预测函数
    def __pred_pm(self, residual_lt: list):
        countp = 0
        countm = 0
        for i in residual_lt:
            if i > 0:
                countp += 1
            else:
                countm += 1
        # 计算正负号概率
        return countp / self.data_shape, countm / self.data_shape

    # 定义拟合函数
    def fit(self) -> np.array:
        # 初始化
        epoch1 = gm11(self.data)
        epoch1_fit = epoch1.fit()
        self.__coff_matrix[0][0], self.__coff_matrix[0][1] = epoch1.coff[0], epoch1.coff[1]
        epoch1_error = epoch1.errors()
        # 初始化符号序列
        pm_lt = self.__count_pm(epoch1_error)
        self.__predict_pm[0][0], self.__predict_pm[0][1] = self.__pred_pm(pm_lt)
        # 获取第一次迭代的拟合值
        sim_data = epoch1_fit
        # 更新权重、背景值系数
        new_weights = self.__update_weight(self.__init_weight, epoch1_error)
        new_background_value = self.__update_background_value(new_weights=new_weights)
        new_data = abs(np.array(epoch1_error))
        new_data = pd.DataFrame(new_data)
        count = 0
        loss = 10
        # 进行迭代
        while count < self.max_baseLearner:  # and loss > 0.1:
            epochs = gm11(new_data, bg_coff=new_background_value)
            # 通过符号序列计算新的拟合值
            sim_data_temp = epochs.fit()
            self.__coff_matrix[count + 1][0], self.__coff_matrix[count + 1][1] = epochs.coff[0], epochs.coff[1]
            for i in range(self.data_shape):
                sim_data[i] = sim_data[i] + pm_lt[i] * sim_data_temp[i]
            # 通过拟合值计算新的差值
            epochs_error = (np.array(self.data).reshape(1, self.data_shape) - sim_data)[0]
            # 通过新的差值更新符号序列
            pm_lt = self.__count_pm(epochs_error)
            # 保存最后n个符号
            # self.__pmpred_mat[count, :] = pm_lt[-285-self.predict_step:-285]
            self.__pmpred_mat[count, :] = pm_lt[-self.a- self.predict_step:-self.a]
            # 进行符号序列预测
            self.__predict_pm[count + 1][0], self.__predict_pm[count + 1][1] = self.__pred_pm(pm_lt)
            # 更新权重、背景值
            new_weights = self.__update_weight(new_weights, epochs_error)
            new_background_value = self.__update_background_value(new_weights)
            new_data = abs(epochs_error)
            new_data = pd.DataFrame(new_data)
            count += 1
            # loss = self.MSE(sim_data)
        return sim_data

    # 定义预测函数
    def predict(self) -> np.array:
        for j in range(self.data_shape + 1, self.data_shape + 1 + self.predict_step):
            # 每一个基学习器的预测值
            temp_sum_lt = []
            count1 = 0
            for i in self.__coff_matrix:
                x = (1 - np.exp(i[0])) * (self.data[0][0] - i[1] / i[0]) * np.exp(-i[0] * (j - 1))
                temp_sum_lt.append(x)
                count1 += 1
            temp_upper_lt = [temp_sum_lt[0]]
            temp_lower_lt = [temp_sum_lt[0]]
            for i in range(1, len(temp_sum_lt)):
                # self.__predict_pm[i - 1][0]
                temp_upper_lt.append(temp_sum_lt[i] * 1)
                temp_lower_lt.append(-temp_sum_lt[i] * 1)
            self.predict_array[j - self.data_shape - 1][0] = sum(temp_upper_lt)
            self.predict_array[j - self.data_shape - 1][2] = sum(temp_lower_lt)
            # 预测以最近的n个符号序列为依据
            temp_val = temp_sum_lt[0]
            for i in range(self.__pmpred_mat.shape[0]):
                temp_val = temp_val + self.__pmpred_mat[i, j - self.data_shape - 1]
            self.predict_array[j - self.data_shape - 1][1] = temp_val
        return self.predict_array

    # 定义损失函数
    def MSE(self, sim_data: np.array) -> float:
        for i in range(self.data_shape):
            self.error.append(round(abs(sim_data[i] - self.data.iloc[i, 0]) / self.data.iloc[i, 0], 8))
        return sum(self.error) / len(self.error)


if __name__ == '__main__':
    import time
    data = pd.read_excel('test.xlsx', sheet_name='Sheet8', header=None)
    start = time.time()
    a = AdaboostGM(data, max_baseLearner=15, target_acc=0.01, predict_step=50, lr=0.2, a=243)
    value = a.fit()
    end = time.time()
    print('运行时间是： ', end-start)
    pre_value = a.predict()
    b = gm11(data, predstep=50)
    valueb = b.fit()
    print('AdaBoostGM的误差是: ', a.MSE(value))
    print('GM(1,1)的误差是： ', b.MSE())
    
    import matplotlib.pyplot as plt
    x = [i for i in range(data.shape[0])]
    x1 = [i for i in range(data.shape[0] - 1, data.shape[0] + 49)]
    plt.plot(x, data, label='Real Data',ls='--')  # ,marker='*')
    plt.plot(x, value, label='GMAdaboost-50')  # ,marker='o')
    plt.title('GMAdaboost-50')
    plt.legend()
    plt.show()
    pre_real_data = pd.read_excel('test.xlsx', sheet_name='Sheet12', header=None).values
    plt.plot(x1, pre_real_data, label='Real Data', ls='--')
    plt.plot(x1, pre_value[:, 1], label='GMAdaboost-15')
    plt.title('GMAdaboost-15')
    plt.legend()
    plt.show()
