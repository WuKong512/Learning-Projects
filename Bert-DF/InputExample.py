import DataPreparation

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, sentence_label=None, aspect_label=None, polarity=None):
        self.guid = guid                        # 输入数据的id
        self.text_a = text_a                    # 输入的句子
        self.text_b = text_b                    # 句子中的aspect
        self.sentence_label = sentence_label    # 句子标注
        self.aspect_label = aspect_label        # text_b的标注
        self.polarity = polarity                # 情感倾向

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


def get_train_examples(train_load):
    train_data = DataPreparation.readfile(train_load)
    train_examples = create_example(train_data, "train")
    return train_examples

def get_test_examples(test_load):
    test_data = DataPreparation.readfile(test_load)
    test_examples = create_example(test_data, "test")
    return test_examples
'''
print(train_examples[0].guid)
print(train_examples[0].text_a)
print(train_examples[0].text_b)
print(train_examples[0].sentence_label)
print(train_examples[0].aspect_label)
print(train_examples[0].polarity)
'''


