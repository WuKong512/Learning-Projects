# 003.Bert-DF

# 002.Bert-z
## Bert
模型Bert通过BertEmbeddings得到包含Token、Segment、Position的输入，在由多层BertLayer组成的Encoder中，得到每个token与其他token的自注意力分数，然后采用gelu激活函数实现非线性功能，Dropout以及归一化处理来防止模型过拟合以及加快训练网络的收敛性。在BertPreTrainingHeads中，得到MLM和NSP下的分值。在 BertForPreTraining中，使用交叉熵损失函数计算score和loss，然后反向传播。
## SSpiders
SSpiders中含有两种爬虫程序，一个是爬取微博的关于某搜索的博客文章，将其保存为excel表格；另一个是实现扫码登录b站以获取视频的评论信息包括ip地址等，将其保存到MongoDB数据库中。

# 001.StudentSystem
