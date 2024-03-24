import tqdm
from transformers import BertTokenizer
import InputExample


model_path = r"C:\Users\51268\.cache\huggingface\hub\models--bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)


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
        text_spc_tokens.extend(aspect_tokens)  # 将输入文本（text_a）和识别出来的实体（text_b）连接起来
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
            for m in range(len(token)):
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
        valid.insert(0, 1)  # 列表插入，成为插入后第b个元素
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


def get_train_features(train_load, label_list, max_seq_length):
    train_examples = InputExample.get_train_examples(train_load)
    train_examples = convert_polarity(train_examples)
    train_features = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer)
    return train_features

def get_test_features(test_load, label_list, max_seq_length):
    test_examples = InputExample.get_test_examples(test_load)
    test_examples = convert_polarity(test_examples)
    test_features = convert_examples_to_features(test_examples, label_list, max_seq_length, tokenizer)
    return test_features


'''
print(train_features[0].input_ids_spc)
print(train_features[0].input_mask)
print(train_features[0].segment_ids)
print(train_features[0].label_id)
print(train_features[0].valid_ids)
print(train_features[0].label_mask)
print(train_features[0].polarities)
'''






