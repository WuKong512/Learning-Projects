def readfile(filename):
    f = open(filename, encoding='utf-8')
    data = []
    sentence = []
    tag = []
    polarity = []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0] =="\n":
            if len(sentence) > 0:
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

    if len(sentence) > 0:
        data.append((sentence, tag, polarity))
    return data

train_data = readfile("./datasets/notebook/notebook.atepc.train.dat")
test_data = readfile("./datasets/notebook/notebook.atepc.test.dat")
'''
print("训练集数量：%d 测试机数量：%d" % (len(train_data),len(test_data)))
print("实例：")
print(train_data[0])
'''