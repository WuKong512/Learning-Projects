# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
from transformers import BertTokenizer, BertModel
from transformers.models.bert.modeling_bert import BertSelfAttention, BertForTokenClassification, BertPooler


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')


# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助



#!pip install pytorch_transformers==1.2.0</code>
import os
import json
import tqdm
import torch
import numpy as np
from transformers.optimization import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset



def readfile(filename):
    f = open(filename, encoding='utf8')
    data = []
    sentence = []
    tag= []
    polarity = []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0 ]=="\n":
            if len(sentence)> 0:
                data.append((sentence, tag, polarity))
                sentence = []
                tag = []
                polarity = []
            continue
        splits = line.split(' ')
        if len(splits) != 3:
            print('warning! detected error line(s) in input file:{}'.format(line))
        sentence.append(splits[0])
        tag.append(splits[-2])
        polarity.append(int(splits[-1][:-1]))

    if len(sentence)> 0:
        data.append((sentence, tag, polarity))
    return data


train_data = readfile("./datasets/notebook/notebook.atepc.train.dat")
test_data = readfile("./datasets/notebook/notebook.atepc.test.dat")

print("训练集数量：%d 测试集数量：%d" % (len(train_data), len(test_data)))
print("实例：")
print(train_data[0])



class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, sentence_label=None, aspect_label=None, polarity=None):
        self.guid = guid  # 输入数据的id
        self.text_a = text_a # 输入的句子
        self.text_b = text_b # 句子中的aspect
        self.sentence_label = sentence_label # 句子标注
        self.aspect_label = aspect_label # text_b的标注
        self.polarity = polarity # 情感倾向

def create_example(lines, set_type):
    examples = []
    for i, (sentence, tag, polarity) in enumerate(lines):
        aspect = []
        aspect_tag = []
        aspect_polarity = [-1]
        for w, t, p in zip(sentence, tag, polarity):
            if p != -1:
                aspect.append(w)
                aspect_tag.append(t)
                aspect_polarity.append(-1)
        guid = "%s-%s" % (set_type, i)
        text_a = sentence
        text_b = aspect
        polarity.extend(aspect_polarity)
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, sentence_label=tag,
                         aspect_label=aspect_tag, polarity=polarity))
    return examples

train_examples = create_example(train_data, "train")
test_examples = create_example(test_data, "test")
print(train_examples[0].guid)
print(train_examples[0].text_a)
print(train_examples[0].text_b)
print(train_examples[0].sentence_label)
print(train_examples[0].aspect_label)
print(train_examples[0].polarity)



MAX_SEQUENCE_LENGTH = 80
LABEL_LIST = ["O", "B-ASP", "I-ASP", "[CLS]", "[SEP]"]
PRETRAINED_BERT_MODEL = "bert-base-chinese"
NUM_LABELS = len(LABEL_LIST) + 1


tokenizer = BertTokenizer.from_pretrained(PRETRAINED_BERT_MODEL, do_lower_case=True)


def convert_polarity(examples):
    for i in range(len(examples)):
        polarities = []
        for polarity in examples[i].polarity:
            if polarity == 2:
                polarities.append(1)
            else:
                polarities.append(polarity)
        examples[i].polarity = polarities
    return examples

train_examples = convert_polarity(train_examples)
test_examples = convert_polarity(test_examples)


class InputFeatures(object):
    def __init__(self, input_ids_spc, input_mask, segment_ids, label_id,
                 polarities=None, valid_ids=None, label_mask=None):
        self.input_ids_spc = input_ids_spc
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.polarities = polarities


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    for example in tqdm.tqdm(examples):
        text_spc_tokens = example.text_a
        aspect_tokens = example.text_b
        sentence_label = example.sentence_label
        aspect_label = example.aspect_label
        polaritiylist = example.polarity
        tokens = []
        labels = []
        polarities = []
        valid = []
        label_mask = []
        text_spc_tokens.extend(['[SEP]'])
        text_spc_tokens.extend(aspect_tokens)  # 将输入文本（text_a）和识别出来的实体(text_b)连接起来
        enum_tokens = text_spc_tokens
        sentence_label.extend(['[SEP]'])
        # sentence_label.extend(['O'])
        sentence_label.extend(aspect_label)
        label_lists = sentence_label
        for i, word in enumerate(enum_tokens):  # 为文本和实体生成标签序列
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = label_lists[i]
            polarity_1 = polaritiylist[i]
            for m in range(len(token)):  # 一个词中不同字，只在首字上标注
                if m == 0:
                    labels.append(label_1)
                    polarities.append(polarity_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            polarities = polarities[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 1)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        input_ids_spc = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids_spc)
        label_mask = [1] * len(label_ids)
        # 将各属性补齐
        while len(input_ids_spc) < max_seq_length:
            input_ids_spc.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        while len(polarities) < max_seq_length:
            polarities.append(-1)
        assert len(input_ids_spc) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        features.append(
            InputFeatures(input_ids_spc=input_ids_spc,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          polarities=polarities,
                          valid_ids=valid,
                          label_mask=label_mask))
    return features

train_features = convert_examples_to_features(train_examples, LABEL_LIST, MAX_SEQUENCE_LENGTH, tokenizer)
test_features = convert_examples_to_features(test_examples, LABEL_LIST, MAX_SEQUENCE_LENGTH, tokenizer)
print(train_features[0].input_ids_spc)
print(train_features[0].input_mask)
print(train_features[0].segment_ids)
print(train_features[0].label_id)
print(train_features[0].valid_ids)
print(train_features[0].label_mask)
print(train_features[0].polarities)


LEARNING_RATE = 3e-5
BATCH_SIZE = 32//16
DEVICE = "cpu"
# DEVICE = "cuda:0"


bert_base_model = BertModel.from_pretrained(PRETRAINED_BERT_MODEL)
bert_base_model.config.num_labels = NUM_LABELS

class SelfAttention(torch.nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_vec = np.zeros((inputs.size(0), 1, 1, MAX_SEQUENCE_LENGTH))
        zero_tensor = torch.tensor(zero_vec).float().to(DEVICE)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])


class Model(BertForTokenClassification):
    def __init__(self, bert_base_model):
        config = bert_base_model.config
        super(Model, self).__init__(config=config)
        self.bert_for_global_context = bert_base_model  # BERT编码器
        self.bert_for_local_context = bert_base_model
        self.pooler = BertPooler(config)  # 池化层
        self.dense = torch.nn.Linear(768, 3)  # 全连接层
        self.bert_global_focus = self.bert_for_global_context
        self.dropout = torch.nn.Dropout(0.1)  # dropout层
        self.SA1 = SelfAttention(config)  # 自注意力机制
        self.SA2 = SelfAttention(config)
        self.linear_double = torch.nn.Linear(768 * 2, 768)  # 全连接层
        self.linear_triple = torch.nn.Linear(768 * 3, 768)

    def get_ids_for_local_context_extractor(self, text_indices):
        text_ids = text_indices.detach().cpu().numpy()
        for text_i in range(len(text_ids)):
            sep_index = np.argmax((text_ids[text_i] == 102))
            text_ids[text_i][sep_index + 1:] = 0
        return torch.tensor(text_ids).to(DEVICE)

    def get_batch_token_labels_bert_base_indices(self, labels):
        if labels is None:
            return
        labels = labels.detach().cpu().numpy()
        for text_i in range(len(labels)):
            sep_index = np.argmax((labels[text_i] == 5))
            labels[text_i][sep_index + 1:] = 0
        return torch.tensor(labels).to(DEVICE)

    def get_batch_polarities(self, b_polarities):
        b_polarities = b_polarities.detach().cpu().numpy()
        shape = b_polarities.shape
        polarities = np.zeros((shape[0]))
        i = 0
        for polarity in b_polarities:
            polarity_idx = np.flatnonzero(polarity + 1)
            polarities[i] = polarity[polarity_idx[0]]
            i += 1
        polarities = torch.from_numpy(polarities).long().to(DEVICE)
        return polarities

    def forward(self, input_ids_spc, token_type_ids=None, attention_mask=None, labels=None, polarities=None, valid_ids=None, attention_mask_label=None):
        input_ids_spc = self.get_ids_for_local_context_extractor(input_ids_spc)
        labels = self.get_batch_token_labels_bert_base_indices(labels)
        global_context_out, _ = self.bert_for_global_context(input_ids_spc, token_type_ids, attention_mask)
        polarity_labels = self.get_batch_polarities(polarities)
        batch_size, max_len, feat_dim = global_context_out.shape
        global_valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32).to(DEVICE)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    global_valid_output[i][jj] = global_context_out[i][j]
        global_context_out = self.dropout(global_valid_output)
        ate_logits = self.classifier(global_context_out)
        pooled_out = self.pooler(global_context_out)
        pooled_out = self.dropout(pooled_out)
        apc_logits = self.dense(pooled_out)
        if labels is not None:
            # 训练过程计算损失
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
            loss_sen = torch.nn.CrossEntropyLoss()
            loss_ate = loss_fct(ate_logits.view(-1, self.num_labels), labels.view(-1))
            loss_apc = loss_sen(apc_logits, polarity_labels)
            return loss_ate, loss_apc
        else:
            return ate_logits, apc_logits

model = Model(bert_base_model)

_ = model.to(DEVICE)


param_optimizer = list(model.named_parameters())  # 模型中的所有参数
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.00001},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.00001}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, weight_decay=0.00001)


all_spc_input_ids = torch.tensor([f.input_ids_spc for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
all_polarities = torch.tensor([f.polarities for f in train_features], dtype=torch.long)
train_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_polarities, all_valid_ids, all_lmask_ids)
train_sampler = SequentialSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

all_spc_input_ids = torch.tensor([f.input_ids_spc for f in test_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
all_polarities = torch.tensor([f.polarities for f in test_features], dtype=torch.long)
all_valid_ids = torch.tensor([f.valid_ids for f in test_features], dtype=torch.long)
all_lmask_ids = torch.tensor([f.label_mask for f in test_features], dtype=torch.long)
eval_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_polarities, all_valid_ids, all_lmask_ids)
eval_sampler = RandomSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=BATCH_SIZE)


import sys
import logging
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report


EPOCH = 5//5  # 共计算5个epoch
EVAL_STEP = 100  # 每10个step执行一个评估

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def evaluate(dataloader, label_list):
    apc_result = {'max_apc_test_acc': 0, 'max_apc_test_f1': 0}
    ate_result = 0
    y_true = []
    y_pred = []
    n_test_correct, n_test_total = 0, 0
    test_apc_logits_all, test_polarities_all = None, None
    model.eval()  # 将网络设置为评估的状态
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    for input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, l_mask in dataloader:
        input_ids_spc = input_ids_spc.to(DEVICE)
        input_mask = input_mask.to(DEVICE)
        segment_ids = segment_ids.to(DEVICE)
        valid_ids = valid_ids.to(DEVICE)
        label_ids = label_ids.to(DEVICE)
        polarities = polarities.to(DEVICE)
        l_mask = l_mask.to(DEVICE)
        with torch.no_grad():
            ate_logits, apc_logits = model(
                input_ids_spc, segment_ids, input_mask,
                valid_ids=valid_ids, polarities=polarities, attention_mask_label=l_mask)
            polarities = model.get_batch_polarities(polarities)
            n_test_correct += (torch.argmax(apc_logits, -1) == polarities).sum().item()
            n_test_total += len(polarities)
            if test_polarities_all is None:
                test_polarities_all = polarities
                test_apc_logits_all = apc_logits
            else:
                test_polarities_all = torch.cat((test_polarities_all, polarities), dim=0)
                test_apc_logits_all = torch.cat((test_apc_logits_all, apc_logits), dim=0)
            label_ids = model.get_batch_token_labels_bert_base_indices(label_ids)
            ate_logits = torch.argmax(F.log_softmax(ate_logits, dim=2), dim=2)
            ate_logits = ate_logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()
            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_ids[i][j] == len(label_list):
                        y_true += temp_1
                        y_pred += temp_2
                        break
                    else:
                        temp_1.append(label_map.get(label_ids[i][j], 'O'))
                        temp_2.append(label_map.get(ate_logits[i][j], 'O'))
    test_acc = n_test_correct / n_test_total
    test_f1 = f1_score(torch.argmax(test_apc_logits_all, -1).cpu(), test_polarities_all.cpu(), labels=[0, 1],
                       average='macro')
    test_acc = round(test_acc * 100, 2)
    test_f1 = round(test_f1 * 100, 2)
    apc_result = {'max_apc_test_acc': test_acc, 'max_apc_test_f1': test_f1}
    report = classification_report(y_true, y_pred, digits=4)
    tmps = report.split()
    ate_result = round(float(tmps[7]) * 100, 2)
    return apc_result, ate_result


max_apc_test_acc = 0
max_apc_test_f1 = 0
max_ate_test_f1 = 0
global_step = 0
for epoch in range(EPOCH):
    # 每个epoch
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # 一个step
        model.train()  # 将网络设置为train的模式
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, l_mask = batch  # 取一个batch的数据
        loss_ate, loss_apc = model(
            input_ids_spc, segment_ids, input_mask, label_ids, polarities, valid_ids, l_mask)  # 前向传播，计算损失
        loss = loss_ate + loss_apc
        loss.backward()  # 反向传播计算梯度
        nb_tr_examples += input_ids_spc.size(0)
        nb_tr_steps += 1
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        if global_step % EVAL_STEP == 0:  # 评估
            apc_result, ate_result = evaluate(eval_dataloader, LABEL_LIST)
            if apc_result['max_apc_test_acc']> max_apc_test_acc:
                max_apc_test_acc = apc_result['max_apc_test_acc']
            if apc_result['max_apc_test_f1']> max_apc_test_f1:
                max_apc_test_f1 = apc_result['max_apc_test_f1']
            if ate_result> max_ate_test_f1:
                max_ate_test_f1 = ate_result
            current_apc_test_acc = apc_result['max_apc_test_acc']
            current_apc_test_f1 = apc_result['max_apc_test_f1']
            current_ate_test_f1 = round(ate_result, 2)
            logger.info('*' * 10)
            logger.info('Epoch %s' % epoch)
            logger.info(f'APC_test_acc: {current_apc_test_acc}(max: {max_apc_test_acc})  '
                        f'APC_test_f1: {current_apc_test_f1}(max: {max_apc_test_f1})')
            logger.info(f'ATE_test_f1: {current_ate_test_f1}(max:{max_ate_test_f1})')
            logger.info('*' * 10)


SAVE_PATH = "./temp/"
os.makedirs(SAVE_PATH, exist_ok=True)
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
label_map = {i : label for i, label in enumerate(LABEL_LIST,1)}
model_config = {
    "bert_model": PRETRAINED_BERT_MODEL,
    "do_lower": True,
    "max_seq_length": MAX_SEQUENCE_LENGTH,
    "num_labels": len(LABEL_LIST)+1,
    "label_map": label_map
}
json.dump(model_config, open(os.path.join(SAVE_PATH, "config.json"), "w"))

LABEL_LIST = ["O", "B-ASP", "I-ASP", "[CLS]", "[SEP]"]