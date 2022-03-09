# 一个基于GM(1,1)模型的改进AdaBoost算法

# An improved AdaBoost algorithm based on GM(1,1)

`greymodel.py`中是GM(1,1)模型

```
使用方法：
1.实例化类     model = gm11(data,predstep=2)
2.训练模型     model.fit()
3.查看拟合误差  model.MSE()
4.预测        model.predict()

Ps:背景值系数bg_coff接收的是一个列表（主要为了后面新模型的构建），默认值为一个空列表，此时背景值系数默认全部为0.5
```

`GMAdaBoost.py`中是提出的改进AdaBoost算法

```
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
```

`test.py`中是论文以及测试使用的数据，其中`Sheet8`是南京大气污染数据，已经使用线性插值补全

| 16   |
| ---- |
| 14   |
| 6    |
| 6    |
| 8    |
| 14   |
| 9    |
| 11   |
| 8    |
| 10   |

