###

以后用
train.py，trainer_ledepth.py，
test_ledepth.py，tester_ledepth.py

trainer.py不用了，trainer_ledepth.py兼容所有
train.py兼容所有，自动识别config里的model然后import里面的LEDepth类

tester.py不用了，里面还aled的指标，以后使用tester_ledepth.py兼容所有并且指标换成esdepth一样。
test.py不用了，test_ledepth.py兼容所有，自动识别config里的model然后import里面的LEDepth类


### aled.py
该模型是基于unet架构，cnn+rnn的融合事件与激光雷达的网络

### delta.py
该模型是基于Transformer的自注意力机制和交叉注意力机制的融合事件与激光雷达的网络，可以理解为将aled的cnn换成注意力机制，另外delta只预测bf不预测af

### LEDepth.py
1、student-teacher：计算了evcod_evt[-1]也就是最后一个SA的输出经过一个aux_sspl_head解码器输出一个辅助深度aux_depth，与原始lidar计算损失，让event作为学生

2、salve-master：经过head得到的encod_lidar与evcod_evt[-1]经过交叉注意力，让event做为qk（slave），让lidar做为v（master），输出结果在经过多个SA，后续和DELTA一样

### LEDepth_SSM
1、将SA替换成VSSM Block，取消了CA，而是直接cat做为fusion

2、每一层的VSSM Block都有一个mem（new block）输出，保存在列表里做为一个储存库，每一次训练都会更新

### LEDepth_SSM_smst
1、salve-master：将简单的cat换成VSSM的交叉版本，利用ht公式，将evt做为▲，做为传播规则，lidar是传播内容
2、student-teacher：还是将evcod进过轻量化解码头解码得到一个深度图然后计算loss

### ledepth_ssm_unet
使用vssm作为encoder和decoder，unet架构，各自编码后cat在一起然后解码，解码的时候跳跃连接上之前编码的lidar和人event

###
