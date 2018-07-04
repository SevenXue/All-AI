# -*- coding:utf-8 -*-
import jieba
import jieba.analyse


jieba.analyse.set_stop_words('stop_words.txt')
jieba.load_userdict("userdict.txt")
jieba.enable_parallel(4)

shop_id = 1142400
content_url = './datasets' + str(shop_id) + '.csv'
output = './dataset/label/' + str(shop_id) + '.csv'
with open(content_url, 'r') as f:
    sms = f.readlines()
    label_dict = {0: [u'签收'], 1: [u'上新', u'新品上架'], 2: [u'快递', u'速递', u'速运', u'查收'], 3: [u'双11', u'双十一',u'双12', u'双十二'], 4: [u'成功付款'],
                  5: [u'没有付款', u'抓紧时间付款', u'点击付款', u'尚未付款', u'快付款', u'还未付款']}
    nums = [0 for i in range(6)]
    distinct = []
    with open(output, 'w') as cf:
        for sm in sms:
            keywords = jieba.analyse.extract_tags(sm, topK=4, withWeight=False, allowPOS=('n', 'vn', 'v'))

            # 进行去重
            if len(distinct) == 0:
                distinct.append(keywords)
                tsp = 2
            else:
                lab_list = []
                for dis in distinct:
                    label = 0
                    for key in keywords:
                        if key not in dis:
                            label += 1
                    lab_list.append(label)
                tsp = min(lab_list)
            if tsp > 1:
                distinct.append(keywords)
                # 进行类别标注
                tmp = 0
                for item in keywords:
                    if tmp == 0:
                        for i in range(6):
                            if item in label_dict[i]:
                                tmp = i
                                if nums[i] == 0:
                                    break
                                else:
                                    cf.write('__label__' + str(i) + '\t' + sm.strip() + '\n')
                                    nums[i] += 1
                                    break
                    if tmp != 0:
                        break
        print nums
