# -*- coding:utf-8 -*-

#对不同店铺的短信进行打标签

import jieba
import jieba.analyse
import time
import re

jieba.analyse.set_stop_words('stop_words.txt')
jieba.load_userdict("userdict.txt")
jieba.enable_parallel(4)

#创建存档文件
shop_id = 2982501
content_url = './datasets/' + str(shop_id) + '.csv'
output = './datasets/label/' + str(shop_id) + '_label.csv'
key_put = './datasets/keywords/' + str(shop_id) + '_keys.csv'
time_put = './datasets/times/' + str(shop_id) + '_times.csv'

#打标签
with open(content_url, 'r') as f:
    sms = f.readlines()
    label_dict = {0: [u'签收'], 1: [u'上新', u'新品上架'], 2: [u'快递', u'速递', u'速运', u'查收'], 3: [u'双11', u'双十一',u'双12', u'双十二'], 4: [u'成功付款'],
                  5: [u'没有付款', u'抓紧时间付款', u'点击付款', u'尚未付款', u'快付款', u'还未付款']}
    nums = [0 for i in range(6)]
    with open(output, 'w') as of:
        for sm in sms:
            keywords = jieba.analyse.extract_tags(sm, topK=4, withWeight=False, allowPOS=('n', 'vn', 'v'))
            tmp = 0
            for item in keywords:
                if tmp == 0:
                    for i in range(6):
                        if item in label_dict[i]:
                            tmp = i
                            of.write(str(i) + ',' + sm.strip() + '\n')
                            nums[i] += 1
                            break
                if tmp != 0:
                    break
    print nums

#分析
with open(output, 'r') as af:
    sms = af.readlines()
    print(len(sms))
    str_0 = ''
    label_str = ['' for i in range(6)]
    time_list = [[0 for k in range(24)] for s in range(6)]
    for sm in sms:
        content = sm.strip().split(',')
        label = int(content[0])
        label_str[label] += content[1]
        send_time = content[-1][1:-1]
        if re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}').match(send_time):
            hour = time.strptime(send_time, '%Y-%m-%d %H:%M:%S').tm_hour
            time_list[label][hour] += 1
    print time_list

    #时间统计
    with open(time_put, 'w') as tf:
        tf.write(str(shop_id) + '\n')
        for item in time_list:
            tmp = [str(item[i]) for i in range(len(item))]
            tf.write(','.join(tmp))
            tf.write('\n')
    #关键字统计
    with open(key_put, 'w') as ky:
        for j in range(len(label_str)):
            keywords = jieba.analyse.extract_tags(label_str[j], topK=10, withWeight=True, allowPOS=('n', 'vn', 'v'))
            print 'Label is : ' + str(j)
            ky.write('label is :' + str(j) + '\n')
            for item in keywords:
                print item[1], item[0]
                frq = int(item[1] * 10 )
                ky.write(item[0].encode('utf-8') + ',' + str(frq) + '\n')