import requests
import csv
import random
from lxml import etree
import time
from time import sleep
import os
import re
import json
import pymongo
from selenium import webdriver
from time import sleep
from PIL import Image
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
# from 动态加载数据处理.chaojiying import Chaojiying_Client
import httpx
import qrcode



class Douyinspiders():
    def __init__(self):
        self.headers = {
            "authority": "s.weibo.com",
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "referer": "https://s.weibo.com/weibo?q=%E8%BA%BA%E5%B9%B3%E4%B8%BB%E4%B9%89&Refer=realtime_weibo",
            #    "sec-ch-ua": "\"Not A(Brand\";v=\"99\", \"Google Chrome\";v=\"121\", \"Chromium\";v=\"121\"",
            "sec-ch-ua": "\"Chromium\";v=\"122\", \"Not(A:Brand\";v=\"24\", \"Microsoft Edge\";v=\"122\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "same-origin",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"
        }
        self.cookies = {
            "SINAGLOBAL": "5284496298762.275.1700631395147",
            #    "SCF": "AgG32P-vuMk0G7HE2WVHRTipjTY6M9DLWQN7RuwbzcP8qUtBhS8lNWCFVxTGlhL9yD2YAlNVu58ZbmOXWMJjenU.",
            #    "UOR": ",,cn.bing.com",
            "ALF": "1712912451",
            "SUB": "_2A25I9RsTDeRhGeNI71QU9SfOyDWIHXVrixLbrDV8PUJbkNANLUz3kW1NSHO2eBYIkJpN_3b6N86BXPML-Q6aSIVJ",
            "SUBP": "0033WrSXqPxfM725Ws9jqgMF55529P9D9WFeR2N0pW7.pOVeV80Yu1mQ5JpX5KMhUgL.Fo-cShqfSK.Ee0.2dJLoI7fWdgvfM.BLB0zLBoUz",
            #    "_s_tentry": "weibo.com",
            "_s_tentry": "-",
            "Apache": "1967185215303.18.1710320828421",
            "ULV": "1710320828445:5:2:1:1967185215303.18.1710320828421:1709300449374",
            #    "webim_unReadCount": "%7B%22time%22%3A1708151973171%2C%22dm_pub_total%22%3A0%2C%22chat_group_client%22%3A0%2C%22chat_group_notice%22%3A0%2C%22allcountNum%22%3A22%2C%22msgbox%22%3A0%7D"
        }

    def run(self, url, video_id, excel_name):
        # 保存到csv文件
        if os.path.exists(excel_name):  # 如果文件存在，不再设置表头
            f = open(excel_name)
            csv_writer = csv.DictWriter(f)
        else:  # 否则，设置csv文件表头
            f = open(excel_name, mode='a', encoding='utf-8-sig', newline='')
            csv_writer = csv.DictWriter(f, fieldnames=['用户名', '时间', '内容', '性别', 'ip', '年龄'])
        # csv_writer.writeheader()  # 写入表头
        '''
        ip_list = []  # ip属地
        text_list = []  # 评论内容
        create_time_list = []  # 评论时间
        user_name_list = []  # 评论者昵称
        user_url_list = []  # 评论者主页链接
        user_unique_id_list = []  # 评论者抖音号
        like_count_list = []  # 点赞数
        cmt_level_list = []  # 评论级别
        '''

        headers = self.headers
        cookies = self.cookies

        url = "https://www.douyin.com/"
        for page in range(1, 51):
            params = {
                "q": "躺平主义",
                "scope": "ori",
                "suball": "1",
                #        "timescope": "custom:2023-01-01:2023-09-20-23",
                "Refer": "g",
                "page": f"{page}"
            }
            response = requests.get(url, headers=headers, cookies=cookies, params=params)
            tree = etree.HTML(response.text)        # 将字符串格式的文件转化为html文档
            div_all = tree.xpath('//*[@id="merge-all-comment-container"]/div/div[3]/div')
            for div in div_all:
                text = div.xpath(
                    './div/div[@class="RHiEl2d8"]/div["YzbzCgxU Q7m2YYn9"]/div[@class="a9uirtCT"]/span/span/span/span/span/span/span/text()')
                content = ''.join(text).strip()
                name = \
                div.xpath('./div/div[@class="RHiEl2d8"]/div["YzbzCgxU Q7m2YYn9"]/div[1]/div[1]/div/a/span/span/span/span/span/span/text()')[0]
                './div/div[@class="RHiEl2d8"]/div["YzbzCgxU Q7m2YYn9"]/div[1]/div[1]/div/a/span/span/span/span/span/span'
                './div/div[2]/div/div[1]/div[1]/div/a/span/span/span/span/span/span'
                '/div/div[2]/div/div[2]/span/span/span/span/span/span/span'
                '/div/div[2]/div/div[2]/span/span/span/span/span/span/span[1]'
                uid = \
                    div.xpath('./div[@class="card"]/div[1]/div[@class="content"]/div[@class="info"]/div[2]/a/@href')[
                        0].split('/')[-1].split('?')[0]
                time = div.xpath('./div[@class="card"]/div[1]/div[@class="content"]/div[@class="from"]/a/text()')[
                    0].strip().replace('今天', '3月13日 ')

                detail_href = f'https://weibo.com/ajax/profile/detail?uid={uid}'
                header = {
                    'Accept': 'application/json, text/plain, */*',
                    #           'Cookie': 'SINAGLOBAL=3347940640910.3564.1641176576939; SCF=AgG32P-vuMk0G7HE2WVHRTipjTY6M9DLWQN7RuwbzcP8qUtBhS8lNWCFVxTGlhL9yD2YAlNVu58ZbmOXWMJjenU.; UOR=,,cn.bing.com; ALF=1710741791; SUB=_2A25I1DxPDeRhGeFN7VMU9i7FzTuIHXVrqDGHrDV8PUJbkNANLWndkW1NQ-jm6YhPtHbwXvOuQRiSMY6r6peyG_gj; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9W5nlUmfQK3AEmwLBzM.9aH15JpX5KMhUgL.FoM0So2fSo54SoM2dJLoI0nLxKqL1hnL1K2LxKBLBonL12BLxKML1hnLBo2LxK-L1K5L1heLxK.LBKqL1K.LxKnL12qL1-eNS7tt; XSRF-TOKEN=m2-iKKPAw1hkjw92cxfXQxmd; _s_tentry=weibo.com; Apache=3640832253999.0356.1708149805311; ULV=1708149805321:52:1:1:3640832253999.0356.1708149805311:1705983452702; wb_view_log_7361560967=1920*10801; WBPSESS=BxPsvZ0MvqraF_gP8H0BPQy8r21uxYYoGNSZfwpVUwGkqewzVP9hv_FLk5jtqk23M9_0gqoLgSHWrNNX4i0BmHTWvaFJI_fOGQXAwWvUW6Em3fUzYUb1rEeFj1Y5ZR7SWpLavVm6cl8nVp8fEuBD_Q==; webim_unReadCount=%7B%22time%22%3A1708151973171%2C%22dm_pub_total%22%3A0%2C%22chat_group_client%22%3A0%2C%22chat_group_notice%22%3A0%2C%22allcountNum%22%3A22%2C%22msgbox%22%3A0%7D',
                    'Cookie': 'SINAGLOBAL=5284496298762.275.1700631395147; ALF=1712912451; SUB=_2A25I9RsTDeRhGeNI71QU9SfOyDWIHXVrixLbrDV8PUJbkNANLUz3kW1NSHO2eBYIkJpN_3b6N86BXPML-Q6aSIVJ; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WFeR2N0pW7.pOVeV80Yu1mQ5JpX5KMhUgL.Fo-cShqfSK.Ee0.2dJLoI7fWdgvfM.BLB0zLBoUz; _s_tentry=-; Apache=1967185215303.18.1710320828421; ULV=1710320828445:5:2:1:1967185215303.18.1710320828421:1709300449374',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                }
                detail = requests.get(detail_href, headers=header)
                data = detail.json()['data']
                try:
                    ip_location = data['ip_location'].replace('IP属地：', '')
                except Exception as e:
                    ip_location = data['location'].replace('IP属地：', '')
                gender = data['gender']
                if gender == 'f':
                    gender = '男'
                else:
                    gender = '女'
                age = random.randint(18, 26)
                dit = {'用户名': name, '时间': time, '内容': content, '性别': gender, 'ip': ip_location, '年龄': age}
                csv_writer.writerow(dit)
                print(dit)
            print(f"第{page}页数据爬取完成")
            sleep(3)

# 注释
'''
# lxml-etree读取文件

from lxml import etree

html = etree.parse('demo01.html', etree.HTMLParser())
print(type(html))  # <class 'lxml.etree._ElementTree'>  返回节点树

# 查找所有 li 节点
rst = html.xpath('//li') #//代表在任意路径下查找节点为li的所有元素
print(type(rst))   # ==><class 'list'>
print(rst)  # ==> [<Element li at 0x133d9e0>, <Element li at 0x133d9b8>, <Element li at 0x133d990>]  找到的所有符合元素的li节点

# 查找li下带有 class 属性值为 one 的元素
rst2 = html.xpath('//li[@class="one"]')
print(type(rst2))  # ==> <class 'list'>
print(rst2)  # ==>[<Element li at 0x35dd9b8>]

# 查找li带有class属性,值为two的元素,下的div元素下的a元素

rst3 = html.xpath('//li[@class="two"]/div/a') # <class 'list'>
rst3 = rst3[0]  #选中res3列表中的第一个元素

print('-------------\n',type(rst3)) # ==><class 'lxml.etree._Element'>
print(rst3.tag)  # ==>输出res3的标签名
print(rst3.text)  # ==> 输出res3中的文本内容

'''

class Blogin(object):
    def get_qrurl(self) -> list:
        """返回qrcode链接以及token"""
        with httpx.Client() as client:
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0'
            }
            url = 'https://passport.bilibili.com/x/passport-login/web/qrcode/generate?source=main-fe-header'
            data = client.get(url=url, headers=headers)
        total_data = data.json()
        qrcode_url = total_data['data']['url']
        qrcode_key = total_data['data']['qrcode_key']
        data = {}
        data['url'] = qrcode_url
        data['qrcode_key'] = qrcode_key
        print(data)
        return data

    def make_qrcode(self, data):
        """制作二维码"""
        qr = qrcode.QRCode(
            version=5,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(data['url'])
        qr.make(fit=True)
        # fill_color和back_color分别控制前景颜色和背景颜色，支持输入RGB色，注意颜色更改可能会导致二维码扫描识别失败
        img = qr.make_image(fill_color="black")
        img.show()

    def sav_cookie(self, data, id):
        """用于储存cookie"""
        try:
            with open(f'./cookie/{id}.json', 'w') as f:
                json.dump(data, f, ensure_ascii=False)
        except FileNotFoundError:
            os.mkdir('./cookie')         # 创建目录
            with open(f'./cookie/{id}.json', 'w') as f:
                json.dump(data, f, ensure_ascii=False)

    def main_run(self):
        """主函数，运行以获取cookie并保存"""
        data = self.get_qrurl()
        token = data['qrcode_key']
        self.make_qrcode(data)
        time.sleep(8)       # 沉睡8秒
        with httpx.Client() as client:
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0'
            }
            url = f"https://passport.bilibili.com/x/passport-login/web/qrcode/poll?qrcode_key={token}&source=main-fe-header"
            data_login = client.get(url=url, headers=headers)  # 请求二维码状态
            data_login = json.loads(data_login.text)
        code = int(data_login['data']['code'])
        print(code)
        if code == 0:
            cookie = dict(client.cookies)
            print(cookie)
            self.sav_cookie(cookie, 'bilibili_cookies')

    def load_cookie(self) -> dict:
        """用于加载cookie"""
        try:
            file = open(f'./cookie/bilibili_cookies.json', 'r')
            cookie = dict(json.load(file))
        except FileNotFoundError:
            msg = '未查询到用户文件，请确认资源完整'
            cookie = 'null'
            print(msg)
        return cookie

    def person(self):
        """获取个人资料"""
        url = 'https://api.bilibili.com/x/web-interface/nav'
        cookie = self.load_cookie()
        with httpx.Client() as client:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE'
            }
            data = client.get(url=url, headers=headers, cookies=cookie)
        data = data.json()
        person_data = data['data']  # 获取个人信息
        user_name = person_data['uname']  # 用户名
        coin_num = str(person_data['money'])  # 硬币数量
        level = str(person_data['level_info']['current_level'])  # 等级
        face = str(person_data['face'])  # 头像链接
        print(person_data)


class BCommentParse(object):
    def __init__(self, base_url):
        self.headers = {
            "authority": "www.bilibili.com",
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "referer": "https://s.weibo.com/weibo?q=%E8%BA%BA%E5%B9%B3%E4%B8%BB%E4%B9%89&Refer=realtime_weibo",
            "sec-ch-ua": "\"Chromium\";v=\"122\", \"Not(A:Brand\";v=\"24\", \"Microsoft Edge\";v=\"122\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "none",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0'
        }
        self.header = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0'
        }
        self.cookies = {

        }
        self.base_url = base_url

    def my_init(self):
        id = self.base_url.split('video/')[-1].split('?')[0]
        if id.startswith('av'):
            id = id.split('av')[-1]
            self.oid = self.get_avid_title(id)
        else:
            self.oid = self.get_avid_title(id, av=False)
        self.set_page()

    def get_avid_title(self, id_number, av=True):
        """
        获取av号以及视频标题
        :param id_number: av/bv号
        :param av: 是否为av号
        :return: av号
        """
        if av == True:
            api = f'https://api.bilibili.com/x/web-interface/view?aid={id_number}'
        else:
            api = f'https://api.bilibili.com/x/web-interface/view?bvid={id_number}'
        r = requests.get(api, headers=self.header)
        _json = json.loads(r.text)
        self.video_title = _json['data']['title']
        avid = _json['data']['aid']
        return avid

    def set_page(self):
        """
        配置数据库
        :return:
        """
        host = '127.0.0.1'
        port = 27017
        myclient = pymongo.MongoClient(host=host, port=port)
        mydb = 'Bilibili'
        sheetname = self.video_title
        db = myclient[mydb]
        self.post = db[sheetname]


    def parse_comment(self):
        self.my_init()
        base_url = f'https://api.bilibili.com/x/v2/reply?jsonp=jsonp&type=1&oid={self.oid}&sort=2'
        n = 0
        url = base_url + '&pn={}'
        bl = Blogin()
        headers = self.header
        cookies = bl.load_cookie()
        try:
            while True:
                r = requests.get(url.format(n), headers=headers, cookies=cookies)
                _json = json.loads(r.text)
                replies = _json.get('data').get('replies')
                item = {}
                n += 1
                if len(replies) != 0:
                    print(f'\033[34;47m--------------------正在爬取{n}页--------------------')
                    for replie in replies:
                        item['user_id'] = replie.get('member').get('mid')  # 用户id
                        item['user_name'] = replie.get('member').get('uname')  # 用户名
                        item['user_sex'] = replie.get('member').get('sex')  # 性别
                        item['user_level'] = replie.get('member').get('level_info').get('current_level')  # 等级
                        vip = replie.get('member').get('vip').get('vipStatus')  # 是否vip
                        if vip == 1:
                            item['user_is_vip'] = 'Y'
                        elif vip == 0:
                            item['user_is_vip'] = 'N'
                        comment_date = replie.get('ctime')  # 评论日期
                        timeArray = time.localtime(comment_date)
                        otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
                        item['apprecate_count'] = replie.get('like')  # 点赞数
                        item['reply_count'] = replie.get('rcount')  # 回复数
                        item['comment_date'] = otherStyleTime  # 评论日期
                        item['location'] = replie.get('reply_control').get('location')  # ip地址
                        item['comment'] = replie.get('content').get('message')  # 评论内容
                        # 判断数据库中有无此文档，也可用于断点续
                        res = self.post.count_documents(item)
                        if res == 0:
                            data = dict(item)
                            self.post.insert(data)
                            print(f'\033[35;46m{item}\033[0m')
                        else:
                            print('\033[31;44m pass\033[0m')
                    time.sleep(0.5)
                else:
                    print(f'\033[31;44m--------------------程序在第{n}页正常退出！--------------------\033[0m')
                    break
        except:
            pass

    '''
        def login(self):
        bro = webdriver.Edge()
        bro.get('https://www.bilibili.com/?spm_id_from=444.41.0.0')
        bro.maximize_window()
        button_login = bro.find_element(By.CLASS_NAME, 'header-login-entry')
        button_login.click()
        sleep(2)
        account = bro.find_element(By.XPATH, '/html/body/div[4]/div/div[4]/div[2]/form/div[1]/input')
        account.send_keys('账号')
        password = bro.find_element(By.XPATH, '/html/body/div[4]/div/div[4]/div[2]/form/div[3]/input')
        password.send_keys('密码')
        sleep(2)
        button_denglu = bro.find_element(By.CLASS_NAME, 'btn_primary')
        button_denglu.click()
        sleep(1)
        # 对当前页面进行截图保存
        # bro.save_screenshot('aa.png')
        # sleep(2)
        # 确定验证码图片对应的左上角和右下角坐标(裁剪对应的区域)
        img_element = bro.find_element(By.CLASS_NAME, 'geetest_widget')
        img_element.screenshot('yzm.png')
        # location=img_element.location#loction_img返回的是字典{x:?,y:?},返回的事x,y坐标
        # size=code_img_ele.size#size返回的也是字典{height:?,width:?},返回的是高度，宽度坐标
        # #将左上角和右下角的坐标放入一个元组里
        # range1=(int(location['x']),int(location['y']),int(location['x']+size['width']),
        #        int(location['y']+size['height']))
        # #至此整张图片区域已经确定
        # sleep(2)
        # i=Image.open('./aa.png')
        # #裁剪后的名字
        # code_img_name = './yzm.png'
        # #开始裁剪
        # frame=i.crop(range1)
        # #裁剪后保存到文件里
        # frame.save(code_img_name)
        sleep(1)
        chaojiying = Chaojiying_Client('账号', '密码', '软件id')  # 用户中心>>软件ID 生成一个替换 96001
        im = open('yzm.png', 'rb').read()  # 本地图片文件路径 来替换 a.jpg 有时WIN系统须要//
        result = chaojiying.PostPic(im, 9004)['pic_str']
        print(result)
        list = result.split('|')
        img_element_half_width = float(img_element.size['width']) / 2
        img_element_half_height = float(img_element.size['height']) / 2
        for li in list:
            x = int(li.split(',')[0])
            y = int(li.split(',')[1])
            action = ActionChains(bro)
            action.move_to_element_with_offset(img_element, int(x - img_element_half_width),
                                               int(y - img_element_half_height)).click().perform()
            # img_element是验证码元素框,x,y是坐标
            sleep(1)
        button_confirm = bro.find_element(By.CLASS_NAME, 'geetest_commit')
        button_confirm.click()
        sleep(3)
        bro.close()
    '''
'''
callback: jQuery1720631904798407396_1605664873948  #经测试可以不传
jsonp: jsonp  #经测试可以不传
pn: 1  #页码标识
type: 1  #所属类型
oid: 248489241	#视频标识,现在确定为视频av号
sort: 2  #所属分类
_: 1605664874976  #当前时间戳，经测试可以不传

test:https://www.bilibili.com/video/BV1T14y1z71G/?spm_id_from=333.337.search-card.all.click&vd_source=26740c69e2a0dacbfe03b2611b0123b3
https://api.bilibili.com/x/v2/reply/wbi/main?oid=786917067&type=1&mode=3&pagination_str=%7B%22offset%22:%22%22%7D&plat=1&seek_rpid=&web_location=1315875&w_rid=2489bd834ba9b4b9cebd6e99ef662a90&wts=1710498425
'''