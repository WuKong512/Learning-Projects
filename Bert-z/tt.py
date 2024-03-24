import torch


# data = [
#         '有生之年',
#         '愿你勇敢',
#         '愿你平安',
#         '愿你没有苦难',
#         '活的简单',
#         '愿你累了倦了有人为你分担',
#         '愿你每个夜晚都会有美梦作伴',
#         '愿你长路漫漫得偿所愿',
#         '愿这世间烦恼从此与你无关',
#         '愿你遇事全部得你心欢',
#         '愿你前程似锦',
#         '不凡此生'
# ]
# # 建立字典 编号 <--> 字 的对应关系
# s = set([i for j in data for i in j])
# print(s)
# print(len(s))
# a = b = [1,2,3]
# a[2]+=3
# print(b)   # [1, 2, 6]
#
# c = d = 5
# d+=3
# print(c)  # 5


'''

修改pytorch的batch_size
def addbatch(data_train,data_test,batchsize):
    """
    设置batch
    :param data_train: 输入
    :param data_test: 标签
    :param batchsize: 一个batch大小
    :return: 设置好batch的数据集
    """
    data = TensorDataset(data_train,data_test)
    data_loader = DataLoader(data, batch_size=batchsize, shuffle=False)#shuffle是是否打乱数据集，可自行设置

    return data_loader

#设置batch
    traindata=addbatch(traininput,trainlabel,1000)#1000为一个batch_size大小为1000，训练集为10000时一个epoch会训练10次。


'''



# import requests
# import csv
# import random
# from lxml import etree
# from time import sleep
#
# f = open('weibo.csv', mode='a', encoding='utf-8-sig', newline='')
# csv_writer = csv.DictWriter(f, fieldnames=['用户名', '时间', '内容', '性别', 'ip', '年龄'])
# # csv_writer.writeheader()  # 写入表头
#
# headers = {
#     "authority": "s.weibo.com",
#     "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
#     "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
#     "referer": "https://s.weibo.com/weibo?q=%E8%BA%BA%E5%B9%B3%E4%B8%BB%E4%B9%89&Refer=realtime_weibo",
# #    "sec-ch-ua": "\"Not A(Brand\";v=\"99\", \"Google Chrome\";v=\"121\", \"Chromium\";v=\"121\"",
#     "sec-ch-ua": "\"Chromium\";v=\"122\", \"Not(A:Brand\";v=\"24\", \"Microsoft Edge\";v=\"122\"",
#     "sec-ch-ua-mobile": "?0",
#     "sec-ch-ua-platform": "\"Windows\"",
#     "sec-fetch-dest": "document",
#     "sec-fetch-mode": "navigate",
#     "sec-fetch-site": "same-origin",
#     "sec-fetch-user": "?1",
#     "upgrade-insecure-requests": "1",
#     "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"
# }
# cookies = {
#     "SINAGLOBAL": "5284496298762.275.1700631395147",
# #    "SCF": "AgG32P-vuMk0G7HE2WVHRTipjTY6M9DLWQN7RuwbzcP8qUtBhS8lNWCFVxTGlhL9yD2YAlNVu58ZbmOXWMJjenU.",
# #    "UOR": ",,cn.bing.com",
#     "ALF": "1712912451",
#     "SUB": "_2A25I9RsTDeRhGeNI71QU9SfOyDWIHXVrixLbrDV8PUJbkNANLUz3kW1NSHO2eBYIkJpN_3b6N86BXPML-Q6aSIVJ",
#     "SUBP": "0033WrSXqPxfM725Ws9jqgMF55529P9D9WFeR2N0pW7.pOVeV80Yu1mQ5JpX5KMhUgL.Fo-cShqfSK.Ee0.2dJLoI7fWdgvfM.BLB0zLBoUz",
# #    "_s_tentry": "weibo.com",
#     "_s_tentry": "-",
#     "Apache": "1967185215303.18.1710320828421",
#     "ULV": "1710320828445:5:2:1:1967185215303.18.1710320828421:1709300449374",
# #    "webim_unReadCount": "%7B%22time%22%3A1708151973171%2C%22dm_pub_total%22%3A0%2C%22chat_group_client%22%3A0%2C%22chat_group_notice%22%3A0%2C%22allcountNum%22%3A22%2C%22msgbox%22%3A0%7D"
# }
# url = "https://s.weibo.com/weibo"
# for page in range(1, 51):
#     params = {
#         "q": "躺平主义",
#         "scope": "ori",
#         "suball": "1",
# #        "timescope": "custom:2023-01-01:2023-09-20-23",
#         "Refer": "g",
#         "page": f"{page}"
#     }
#     response = requests.get(url, headers=headers, cookies=cookies, params=params)
#     tree = etree.HTML(response.text)
#     div_all = tree.xpath('//*[@id="pl_feedlist_index"]/div[2]/div')
#     for div in div_all:
#         title = div.xpath('./div[@class="card"]/div[1]/div[@class="content"]/p[@node-type="feed_list_content"]/text()')
#         content = ''.join(title).strip()
#         name = div.xpath('./div[@class="card"]/div[1]/div[@class="content"]/div[@class="info"]/div[2]/a/text()')[0]
#         uid = \
#         div.xpath('./div[@class="card"]/div[1]/div[@class="content"]/div[@class="info"]/div[2]/a/@href')[0].split('/')[
#             -1].split('?')[0]
#         time = div.xpath('./div[@class="card"]/div[1]/div[@class="content"]/div[@class="from"]/a/text()')[
#             0].strip().replace('今天', '3月13日 ')
#
#         detail_href = f'https://weibo.com/ajax/profile/detail?uid={uid}'
#         header = {
#             'Accept': 'application/json, text/plain, */*',
# #           'Cookie': 'SINAGLOBAL=3347940640910.3564.1641176576939; SCF=AgG32P-vuMk0G7HE2WVHRTipjTY6M9DLWQN7RuwbzcP8qUtBhS8lNWCFVxTGlhL9yD2YAlNVu58ZbmOXWMJjenU.; UOR=,,cn.bing.com; ALF=1710741791; SUB=_2A25I1DxPDeRhGeFN7VMU9i7FzTuIHXVrqDGHrDV8PUJbkNANLWndkW1NQ-jm6YhPtHbwXvOuQRiSMY6r6peyG_gj; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9W5nlUmfQK3AEmwLBzM.9aH15JpX5KMhUgL.FoM0So2fSo54SoM2dJLoI0nLxKqL1hnL1K2LxKBLBonL12BLxKML1hnLBo2LxK-L1K5L1heLxK.LBKqL1K.LxKnL12qL1-eNS7tt; XSRF-TOKEN=m2-iKKPAw1hkjw92cxfXQxmd; _s_tentry=weibo.com; Apache=3640832253999.0356.1708149805311; ULV=1708149805321:52:1:1:3640832253999.0356.1708149805311:1705983452702; wb_view_log_7361560967=1920*10801; WBPSESS=BxPsvZ0MvqraF_gP8H0BPQy8r21uxYYoGNSZfwpVUwGkqewzVP9hv_FLk5jtqk23M9_0gqoLgSHWrNNX4i0BmHTWvaFJI_fOGQXAwWvUW6Em3fUzYUb1rEeFj1Y5ZR7SWpLavVm6cl8nVp8fEuBD_Q==; webim_unReadCount=%7B%22time%22%3A1708151973171%2C%22dm_pub_total%22%3A0%2C%22chat_group_client%22%3A0%2C%22chat_group_notice%22%3A0%2C%22allcountNum%22%3A22%2C%22msgbox%22%3A0%7D',
#             'Cookie': 'SINAGLOBAL=5284496298762.275.1700631395147; ALF=1712912451; SUB=_2A25I9RsTDeRhGeNI71QU9SfOyDWIHXVrixLbrDV8PUJbkNANLUz3kW1NSHO2eBYIkJpN_3b6N86BXPML-Q6aSIVJ; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WFeR2N0pW7.pOVeV80Yu1mQ5JpX5KMhUgL.Fo-cShqfSK.Ee0.2dJLoI7fWdgvfM.BLB0zLBoUz; _s_tentry=-; Apache=1967185215303.18.1710320828421; ULV=1710320828445:5:2:1:1967185215303.18.1710320828421:1709300449374',
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
#         }
#         detail = requests.get(detail_href, headers=header)
#         data = detail.json()['data']
#         try:
#             ip_location = data['ip_location'].replace('IP属地：', '')
#         except Exception as e:
#             ip_location = data['location'].replace('IP属地：', '')
#         gender = data['gender']
#         if gender == 'f':
#             gender = '男'
#         else:
#             gender = '女'
#         age = random.randint(18, 26)
#         dit = {'用户名': name, '时间': time, '内容': content, '性别': gender, 'ip': ip_location, '年龄': age}
#         csv_writer.writerow(dit)
#         print(dit)
#     print(f"第{page}页数据爬取完成")
#     sleep(3)

# 2024-2-17 15:56


# x = torch.randn(1)
# y = x
# y1 = x**2
#
# print(x)
# print(y)
# print(y1)
# print("---------------")
# y2 = x.pow(2)
# print(y2)
# print(y)
# print(x)
# print("---------------")
# y3 = x.pow_(2)
# print(y3)
# print(y)
# print(x)


# a = [0,1,2,3,4,5,6]
# print(a[-1])
# print(a[-2])

'''
# 观察与探索，寻找规律，想方法。
# 将下面cookies(string)转化为dict
s = "buvid3=566727B7-CA8F-590D-7F54-F31244AE981540712infoc; b_nut=1695205640; i-wanna-go-back=-1; b_ut=7; _uuid=E7762F107-31025-10BA6-E46F-9A1010B108A4104539970infoc; buvid4=1A8F7F58-2D59-4375-E846-C9AC78726CBB41940-023092018-vXCzxafnl%2Fro0JsoMJW7rg%3D%3D; DedeUserID=499696927; DedeUserID__ckMd5=4e63f5dc027c969b; header_theme_version=CLOSE; CURRENT_FNVAL=4048; rpdid=|(J|~YRu~)~k0J'uYmlk|lY~l; PVID=1; CURRENT_QUALITY=120; home_feed_column=5; browser_resolution=1488-742; buvid_fp_plain=undefined; hit-dyn-v2=1; enable_web_push=DISABLE; FEED_LIVE_VERSION=V8; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MTA3NTQ3ODIsImlhdCI6MTcxMDQ5NTUyMiwicGx0IjotMX0.ru-Rt0acxERk0IR1mIeB7Nh8XYX218glZadqWV9_o4A; bili_ticket_expires=1710754722; SESSDATA=9f63d7bf%2C1726291810%2C1ee4f%2A31CjDm-OlZFuoh1oCujKW-55MdHmFhHWTmJSSQyZ8P7qdCDlcA959OtX_lYx4ld1uIpMgSVkNWejNpblNUTG9ma21nQWJ2QXhJLWp3Y0Q0TjJjY1lscWx2QUhiWVZiR3NBN3ZZS240eTZMSTlxSmY4YWZhYWdEM2xyU2NKbjlpeUh6TTdIcnh0QU13IIEC; bili_jct=3e152ce1a08f91234512e42b74b9757b; sid=7ry9s3ff; fingerprint=c155413e837425921d770fd1ae0dbe9a; buvid_fp=c155413e837425921d770fd1ae0dbe9a; bp_video_offset_499696927=910134420798701589; b_lsid=2579810AB_18E50AF1614"
s1 = s.replace('=',':')
s2 = s1.replace(';','\n')
s3 = s.replace('=','\':\'')
s3 = s3.replace(';','\'')
s3 = s3.replace(' ',',\n\'')

print(s3)
'''


import SSpiders

base_url = 'https://www.bilibili.com/video/av323635212'
bilibili = SSpiders.BCommentParse(base_url)
bilibili.parse_comment()        # 运行爬虫
cookie  = SSpiders.Blogin()
cookie.main_run()

