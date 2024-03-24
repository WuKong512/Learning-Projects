# 1.头文件
import copy
import math
import os.path
import shutil
import sys
import logging
import tarfile
import tempfile

import torch
import torch.nn as nn
import torch.utils.data as Data
from random import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# bias 偏置

# 2.设置的参数
class BertConfig():
    def __init__(self):
        self.attention_probs_dropout_prob = 0.1            # 注意力处dropout值
        self.hidden_act = "gelu"                           # 隐藏层使用的激活函数
        self.hidden_dropout_prob = 0.1                     # 隐藏层处dropout值
        self.hidden_size = 1024                            # 隐藏层大小，字向量长度 1024
        self.initializer_range = 0.02                      # bert模型初始化方差值
        self.intermediate_size = 1024                      # 向前传播隐藏层大小 4096
        self.max_position_embeddings = 512                 # 位置信息长度512
        self.num_attention_heads = 16                      # 注意力头的个数 16
        self.num_hidden_layers = 12                        # encoder层数  24
        self.type_vocab_size = 2                           # 句子类型，标记第一句话和第二句话
        self.vocab_size = 21128                            # 字典大小21128
        self.seq_length = 32                               # tokens总长度  40(640)  32(512)

Batch_size  = 32        # 32
Learnrate = 5e-5        # 1e-4
Epochs = 4             # 10 随着epoch 数量的增加， 权重更新迭代的次数增多， 曲线从最开始的不拟合状态， 进入优化拟合状态， 最终进入过拟合。epoch如何设置： 大小与数据集的多样化程度有关， 多样化程度越强， epoch 越大。

# 3.数据的预处理
config = BertConfig()
data = [
        '有生之年',
        '愿你勇敢',
        '愿你平安',
        '愿你没有苦难',
        '活的简单',
        '愿你累了倦了有人为你分担',
        '愿你每个夜晚都会有美梦作伴',
        '愿你长路漫漫得偿所愿',
        '愿这世间烦恼从此与你无关',
        '愿你遇事全部得你心欢',
        '愿你前程似锦',
        '不凡此生'
]
# 建立字典 编号 <--> 字 的对应关系
s = set([i for j in data for i in j])    # for j in data, for i in j,set(i) 集合
                                # 字典大小
word2idx = {'PAD': 0,'CLS': 1,'SEP': 2,'MASK':3}         # 特殊字符
for idx,word in enumerate(s):            # idx为集合中的序号（0始） word为字
    word2idx[word] = idx+4                               # 字  -> 编号
idx2word = {word2idx[key]:key for key in word2idx}       # 编号 -> 字      ,key为键
vocab_size = len(idx2word)

# 把句子的字变成编号
sentences = []
for sentence in data:
    tmp = []
    for i,word in enumerate(sentence):
        tmp.append(word2idx[word])
    sentences.append(tmp)

# 自定义Dataset
batch_size  = Batch_size
class MyDataset(Data.Dataset):
    def __init__(self,data):
        self.sentences = []
        for sentence in data:
            tmp = []
            for i,word in enumerate(sentence):
                tmp.append(word2idx[word])
            self.sentences.append(tmp)
        self.sentences_len = len(self.sentences)
    def __len__(self):
        return len(self.sentences)*2-2
    def __getitem__(self, idx):
        sentences = copy.deepcopy(self.sentences)
        input_ids = []
        token_type_ids = []
        next_sentence_label = []

        if idx%2 == 0:
            s = [word2idx['CLS']] + sentences[int(idx//2)] + [word2idx['SEP']] + sentences[int(idx//2)+1] + [word2idx['SEP']]
            input_ids = s + [0]*(config.seq_length-len(s))
            token_type_ids = [0]*(1+len(sentences[int(idx//2)])+1) + [1]*(len(sentences[int(idx//2+1)])+1) + [0]*(config.seq_length-len(s))
            next_sentence_label = [1]
        else:
            rand = int(idx//2)+1
            while rand == idx//2+1:
                rand = randint(0, self.sentences_len-1)
            s = [word2idx['CLS']] + sentences[int(idx//2)] + [word2idx['SEP']] + sentences[rand] + [word2idx['SEP']]
            input_ids = s + [0]*(config.seq_length-len(s))
            token_type_ids = [0]*(1+len(sentences[int(idx//2)])+1) + [1]*(len(sentences[rand])+1) + [0]*(config.seq_length-len(s))
            next_sentence_label = [0]

        attention_mask = []
        masked_lm_labels = []
        for pos,value in enumerate(input_ids):
            rand = random()
            if value == 0:
                attention_mask.append(0)
            else:
                attention_mask.append(1)
            if value != 0 and value != 1 and value != 2 and rand < 0.15:
                masked_lm_labels.append(input_ids[pos])
                if rand < 0.15*0.8:
                    input_ids[pos] = word2idx['MASK']
                elif rand > 0.15*0.9:
                    input_ids[pos] = randint(4,vocab_size-1)
            else:
                masked_lm_labels.append(-1)
        input_ids = torch.tensor(input_ids)
        token_type_ids = torch.tensor(token_type_ids)
        attention_mask = torch.tensor(attention_mask)
        masked_lm_labels = torch.tensor(masked_lm_labels)
        next_sentence_label = torch.tensor(next_sentence_label)
        return input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_label
data_loader = Data.DataLoader(MyDataset(data),batch_size,True)
# data_loader = DataLoader(data, batch_size=batchsize, shuffle=False)#shuffle是是否打乱数据集，可自行设置

'''
数据预处是采用动态Mask和动态下句随机匹配，每一次迭代，epoch的数据都不同。

input_ids : 句子的字典索引
token_type_ids：用1和0区分第一个句子和第二个句子
attention_mask：标记句子中选出15%的token。
masked_lm_labels:标记被mask的真实字典索引值。
next_sentence_label：标记两句话是否连续
'''


# 4.定义激活函数和归一化
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, 'relu': torch.nn.functional.relu, 'swish': swish}

class BertLayerNorm(nn.Module):         # 归一化，使梯度下降更均匀，收敛更快。
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))         # torch.ones(*size, out=None, dtype=None)返回一个每个元素都是1、形状为size、数据类型为dtype的Tensor。
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self,x):
        u = x.mean(-1, keepdim = True)                     # 最后一个维度上(1024)求均值，可以理解在字向量上
        s = (x - u).pow(2).mean(-1, keepdim = True)        # 最后一个维度上求均值，可以理解在字向量上  pow(z,y,z),指数函数，底，指，余数  这里是(x - u)的平方
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            # 减去一个统计量（最后一个维度的均值），看作以那个点为原点；除以一个统计量对特征进行放缩
        return self.weight * x + self.bias


# 5. BertEmbeddings
# Bert的输入
class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)       # 正则化，防止过拟合。 随机忽略10%的特征

    def forward(self, input_ids, token_type_ids = None):           # input_ids:(batch, seq_length)    token_type_ids:(batch, seq_length)
       seq_length = input_ids.size(1)                              # 表示第1维的数据数量（维度）  seq_length: 40
       position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)    # (seq_length)  arange,返回一维张量，其值取于[start, end)之间，以step为步长
       position_ids = position_ids.unsqueeze(0).expand_as(input_ids)                         # (batch, seq_length)  expand_as(*sizes) 将张量tensor扩展为和参数sizes一样的大小
       if token_type_ids is None:
           token_type_ids = torch.zeros_like(input_ids)
       words_embeddings = self.word_embeddings(input_ids)                                    # (batch, seq_length, hidden_sizek)
       position_embeddings = self.position_embeddings(position_ids)                          # (batch, seq_length, hidden_sizek)
       token_type_embeddings = self.token_type_embeddings(token_type_ids)                    # (batch, seq_length, hidden_sizek)
       embeddings = words_embeddings + position_embeddings + token_type_embeddings
       embeddings = self.LayerNorm(embeddings)                                               # 最后一个维度求归一化
       embeddings = self.dropout(embeddings)
       return embeddings                                                                     # (batch, seq_length, hidden_sizek)

'''
Token Embeddings：是数据预处理后语句的字典索引。
Segment Embeddings：标记哪些是第一句话，那哪是第二句话。
Position Embeddings：标记这个token在这句话处的位置。
把它们映射到维度相同高维空间，再加起来，input = Token Embeddings + Segment Embeddings + Position Embeddings，得到Encoder的输入。
'''

# 6. Bert Encoder
# 6.1 Encoder的Self-Attention Mechanism
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads                               # num_attention_heads个注意力
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)     # 每个注意力大小1024/16=64
        self.all_head_size = self.num_attention_heads * self.attention_head_size            # all_head_size: 16*64=1024
        self.query = nn.Linear(config.hidden_size, self.all_head_size)                      # (hidden_size, attention_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)                        # (hidden_size, attention_head_size
        self.value = nn.Linear(config.hidden_size, self.all_head_size)                      # (hidden_size, attention_head_size
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self,x):                   # q, k, v 改变形状
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)      # [:-1],表示维度0到最后一个维度之前的维度
                                                                                    # new_x_shape:元组(batch, seq_len, num_attention_heads, attention_head_sizek)
        x = x.view(*new_x_shape)                                                    # (batch, seq_len, num_attention_heads, attention_head_size)
        return x.permute(0,2,1,3)                                                   # (batch, num_attention_heads, seq_len, attention_head_size)  将tensor的维度换位。

    def forward(self, hidden_states, attention_mask):                               # (batch, seq_length, hidden_size),(batch,1,1,sqe_len)
        mixed_query_layer = self.query(hidden_states)                               # (batch, seq_len, hidden_size)
        mixed_key_layer = self.key(hidden_states)                                   # (batch, seq_len, hidden_size)
        mixed_value_layer = self.value(hidden_states)                               # (batch, seq_len, hidden_size)

        query_layer = self.transpose_for_scores(mixed_query_layer)                  # (batch, num_attention_heads, seq_len, attention_head_size)
        key_layer = self.transpose_for_scores(mixed_key_layer)                      # (batch, num_attention_heads, seq_len, attention_head_size)
        value_layer = self.transpose_for_scores(mixed_value_layer)                  # (batch, num_attention_heads, seq_len, attention_head_size)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,-2))    # (batch, num_attention_heads, seq_len, seq_len)  transpose()一次只能在两个维度间进行转置（也可以理解为维度转换）
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)   # (batch, num_attention_heads, seq_len, seq_len)
        attention_scores = attention_scores + attention_mask                        # (batch, num_attention_heads, seq_len, seq_len)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)                      # (batch, num_attention_heads, seq_len, seq_len)  对最高维度的行进行softmax运算，和为1  得到注意力分数
        attention_probs = self.dropout(attention_probs)                             # (batch, num_attention_heads, seq_len, seq_len)

        context_layer = torch.matmul(attention_probs, value_layer)                  # (batch, num_attention_heads, seq_len, attention_head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()              # (batch, seq_len, num_attention_heads, attention_head_size)  contiguous(),保证连续。如无，我们在 transpose、permute 操作后执行 view，Pytorch 会抛出错误
#        cl = context_layer.size()[:-2]                                              # cl: torch.Size([16, 40])
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # (batch, seq_len, hidden_size)
        context_layer = context_layer.view(*new_context_layer_shape)                # (batch, seq_len, hidden_size)
        return context_layer                                                        # (batch, seq_len, hidden_size)

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):                     # (batch, seq_len, hidden_size),(batch, seq_len, hidden_sizeK)
        hidden_states = self.dense(hidden_states)                       # (batch, seq_len, hidden_size)
        hidden_states = self.dropout(hidden_states)                     # (batch, seq_len, hidden_size)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)    # (batch, seq_len, hidden_size)
        return hidden_states                                            # (batch, seq_len, hidden_size)

'''
等形线性变换，Dropout正则化，Layer Norm 归一化

Layer Normalization和Batch Normalization的区别：Layer Normalization：是把Batch中每一句话进行归一化。B
atch Normalization：是把每个Batch中每句话的第一个字看成一组做归一化。
'''

class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):    # hidden_states:(batch, seq_length, hidden_size), attention_mask:(batch,1,1,sqe_len)
        self_output = self.self(input_tensor, attention_mask)       # (batch, seq_len, hidden_size)
        attention_output = self.output(self_output, input_tensor)   # (batch, seq_len, hidden_size)
        return attention_output                                     # (batch, seq_len, hidden_size)

'''
可以理解成：先计算出第一个token与句子中的每一个token的注意力分数（包括第一个token），
再用计算出的注意力分数乘以对应token的信息，
然后加在一起，得到的结果就是第一个token与句子中所有token的加权和信息，依次更新每一个token与句子的注意力信息。
'''


# 6.2 Encoder的Add & Layer normalization
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):                               # (batch, seq_len, hidden_size)
        hidden_states = self.dense(hidden_states)                   # (batch, seq_len, intermediate_size
        hidden_states = self.intermediate_act_fn(hidden_states)     # (batch, seq_len, intermediate_size)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):                     # (batch, seq_len, intermediate_size)
        hidden_states = self.dense(hidden_states)                       # (batch, seq_len, hidden_size)
        hidden_states = self.dropout(hidden_states)                     # (batch, seq_len, hidden_size)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)    # (batch, seq_len, hidden_size)
        return hidden_states                                            # (batch, seq_len, hidden_size)

class BertLayer(nn.Module):         # BertLayer是Bert预训练模型中产生句向量和词向量的核心模块，
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):   # hidden_states:(batch, seq_length, hidden_size), attention_mask:(batch,1,1,sqe_len)
        attention_output = self.attention(hidden_states, attention_mask)    # (batch, seq_len, hidden_size)
        intermediate_output = self.intermediate(attention_output)           # (batch, seq_len, intermediate_size)
        layer_output = self.output(intermediate_output, attention_output)   # (batch, seq_len, hidden_size)
        return layer_output                                                 # (batch, seq_len, hidden_size)

'''利用了残差网络可能增加网络深度性，归一化可以加快网络收敛。'''


# 6.3 Encoder
class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):   # hidden_states:(batch, seq_length, hidden_size), attention_mask:(batch,1,1,sqe_len)
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)     # (batch, seq_len, hidden_size)
            if output_all_encoded_layers:                                   # 输出每一层的内容
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:                                   # 输出最后一层的内容
            all_encoder_layers.append(hidden_states)                        # (batch, seq_len, hidden_size)
        return all_encoder_layers                                           # (batch, seq_len, hidden_size)

'''这里组合成24层的Encoder'''


# 9.BertPooler（处理CLS信息）
class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]            # (batch, head_size)  x[:,n]表示在全部数组（维）中取第n个数据，直观来说，x[:,n]就是取所有集合或维度的第n个数据；x[n,:]表示在n个数组（维）中取全部数据，直观来说，x[n,:]就是取第n集合的所有数据,
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

'''获取Encoder最后一层输出的第一个token，也就是"CLS"的编码后的特征信息。'''


# 10 预训练模型加载，保存
class BertPreTrainedModel(nn.Module):
    # 处理权重和下载、加载模型
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
            ))
        self.config = config

    def init_bert_weights(self, module):
        # 初始化权重
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        # 预先训练的模型下载并缓存预先训练的模型文件
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        cache_dir = kwargs.get('cache_dir', None)
        kwargs.pop('cache_dir', None)
        from_tf = kwargs.get('from_tf', False)
        kwargs.pop('from_tf', None)

        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()), archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        config_file = os.path.join(serialization_dir, CONFIG_NAME)  # 加载config文件
        if not os.path.exists(config_file):
            config_file = os.path.join(serialization_dir, BERT_CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu')
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_keys = None
            if 'gamma' in key:
                new_keys = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_keys = key.replace('beta', 'bias')
            if new_keys:
                old_keys.append(key)
                new_keys.append(new_keys)
        for old_keys, new_keys in zip(old_keys, new_keys):
            state_dict[new_keys] = state_dict.pop(old_keys)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_form_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
            start_prefix = ''
            if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
                start_prefix = 'bert.'
            load(model, prefix=start_prefix)
            if len(missing_keys) > 0:
                logger.info("weights of {} not initialized from pretrained model: {}".format(model.__class__.__name__,
                                                                                             missing_keys))
            if len(unexpected_keys) > 0:
                logger.info("weights from pretrained model not used in {}: {}".format(model.__class__.__name__,
                                                                                      unexpected_keys))
            if len(error_msgs) > 0:
                raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(model.__class__.__name__,
                                                                                         "\n\t".join(error_msgs)))
            return model


# 11 BertModel
class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)                          # (batch,1,1,sqe_len)  unsqueeze(),函数起升维的作用,参数表示在哪个地方加一个维度。
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)   # (batch,1,1,sqe_len)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0                        # 只计算被标记的位置(batch,1,1,sqe_len)
        embeddings_output = self.embeddings(input_ids, token_type_ids)                              # (batch, seq_length, hidden_size)
        encoded_layers = self.encoder(embeddings_output,                                            # (batch, seq_length, hidden_size)
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]                                                        # 取出encoder最后一层输出(batch, seq_length, hidden_size)
        pooled_output = self.pooler(sequence_output)                                                # 返回CLS的特征(batch, hidden_size)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output                # 返回最后一次的数据：(batch, seq_length, hidden_size),返回CLS的特征(batch, hidden_size)

'''把前面定义的类组合成Bert'''


# 11 预训练BertPreTrainingHeads（可以理解成bert接的下游任务）
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):                                                 # bert输出(batch, seq_length, hidden_size)
        hidden_states = self.dense(hidden_states)                       # bert输出(batch, seq_length, hidden_size)
        hidden_states = self.transform_act_fn(hidden_states)            # 激活函数
        hidden_states = self.LayerNorm(hidden_states)                   # 归一化
        return hidden_states                                            # (batch, seq_length, hidden_size)

class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0),
            bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):                                   # bert输出(batch, seq_length, hidden_szie)
        hidden_states = self.transform(hidden_states)                   # (batch, seq_length, hidden_size)
        hidden_states = self.decoder(hidden_states) + self.bias         # (batch, seq_length, vocab_size)
        return hidden_states                                            # (batch, seq_length, vocab_size)

class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embeddings_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embeddings_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):                  # bert输出(batch, seq_length, hiddzen_size),CLS的特征(batch, hidden_size)
        prediction_scores = self.predictions(sequence_output)           # (batch, seq_length, vocab_size)
        seq_relationship_scores = self.seq_relationship(pooled_output)  # (batch, 2)
        return prediction_scores, seq_relationship_scores               # (batch, seq_length, vocab_szie), (batch, 2)

'''
对预训练做准备，把"CLS"转换成大小为（batch, 2）。
把Bert整个输出转换成大小为(batch, seq_length,vocab_size)，方便后面做损失。
'''


# 13 预训练模型定义
class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # 返回encoder最后一层(batch, seq_length, hidden_size),返回CLS的特征(batch, hidden_size)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)    # (batch, seq_length, vocab_size), (batch, 2)
        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)                                     # 忽略标签为-1的loss  CrossEntropyLoss 交叉熵损失函数，用于分类任务
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
                                                                                                # 计算mask的loss的平均值  view(-1),张量变成一维；view(-1, a)，-1会根据a的值变以保证“-1*a”的总数不变
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
                                                                                                # 计算两句话是否连续
            total_loss = masked_lm_loss + next_sentence_loss                                    # 两个loss加起来
            return total_loss
        else:
            return prediction_scores, seq_relationship_score                                    # (batch, seq_length, bocab_size), (batch, 2)
config = BertConfig()
model = BertForPreTraining(config).to(device)
learnrate = Learnrate
optimizer = torch.optim.Adam(model.parameters(), lr=learnrate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learnrate, momentum=0.99)      两种优化算法。

'''
定义模型，计算损失：第一部分masked_lm_loss 计算被mask的token，第二部分next_sentence_loss 计算是否两句话是连续，然后加在一起做返回。
total_loss = masked_lm_loss + next_sentence_loss 。

注意nn.CrossEntropyLoss(ignore_index=-1)是忽略标签为-1的损失。
'''


# 14 训练
epochs = Epochs
model.load_state_dict(torch.load('model.params'))     # 加载模型参数
for epoch in range(epochs):
    for input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_label in data_loader:
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        masked_lm_labels = masked_lm_labels.to(device)
        next_sentence_label = next_sentence_label.to(device)
        loss = model(input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_label)
        # 反向传播
        optimizer.zero_grad()                   # 梯度归零
        loss.backward()                         # 反向传播计算每个参数的梯度值
        optimizer.step()                        # 最后通过梯度下降执行一步参数更新
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

torch.save(model.state_dict(), 'model.params')          # 保存模型参数







