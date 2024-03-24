# 003.Bert-DF
## 数据准备
DataPreparation，读入原始数据，转换成文本序列和标记序列。其中标注序列有两个，一个是实体的标注序列，通过O、B-ASP、I-ASP标注出文本中的实体，另一个是情感的标注序列，通过0、-1、2标注出对应实体的情感倾向性。   

InputFeature，将数据处理成BERT模型可以接受的形式，将InputExample中的属性补齐到相同的长度，并以InputFeature的形式作为神经网络的输入。按照BERT的输入格式，text_a和text_b之间通过'[SEP]'符号连接。
## ModelConstruct
先在BERT的基础上实现一个新的自注意力机制。构造网络结构，BertForTokenClassification类完成了基本的token分类的功能，在这个类的基础上搭建模型，覆盖里面的forward方法。forward方法定义了神经网络前向传播的过程。   

构造DataLoader，首先将InputFeature中的特征转化为pytorch中的Tensor，然后生成TensorDataset和SequentialSampler，再将dataset和sampler结合成为DataLoader，为之后模型训练和测试提供数据。
## ModelTraining
定义模型训练以及超参数，定义评估函数evaluate。加载预训练模型bert-base-chinese，设置优化器，使用学习率衰减的策略。训练模型，在每个EPOCH中，每次输入一个batch的数据，计算损失和损失反向传播完成一个step的训练。每经过EVAL_STEP个step做一次评估，并记录下训练过程中最好的评价指标。

保存模型。

# 002.Bert-z
## Bert
模型Bert通过BertEmbeddings得到包含Token、Segment、Position的输入，在由多层BertLayer组成的Encoder中，得到每个token与其他token的自注意力分数，然后采用gelu激活函数实现非线性功能，Dropout以及归一化处理来防止模型过拟合以及加快训练网络的收敛性。在BertPreTrainingHeads中，得到MLM和NSP下的分值。在 BertForPreTraining中，使用交叉熵损失函数计算score和loss，然后反向传播。
## SSpiders
SSpiders中含有两种爬虫程序，一个是爬取微博的关于某搜索的博客文章，将其保存为excel表格；另一个是实现扫码登录b站以获取视频的评论信息包括ip地址等，将其保存到MongoDB数据库中。

# 001.StudentSystem
