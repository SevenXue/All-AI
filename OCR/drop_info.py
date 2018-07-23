from aip import AipOcr
import jieba
import re
import sys
import configparser
import os

config = configparser.ConfigParser()
config.read('drop.ini')

APP_ID = config.get('OCR','APP_ID')
API_KEY = config.get('OCR', 'API_KEY')
SECRET_KEY = config.get('OCR', 'SECRET_KEY')
client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

jieba.add_word(u'款号')
jieba.add_word(u'面料成分')
jieba.add_word(u'成份')
jieba.add_word(u'货号')

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

def get_ingredient(infos):
    """
        提取出面料成分
    Args:
         infos:文本信息
    Returns:
        ingredient:面料成分
    """
    keys = config.get('KEYS','ingers').split(',')
    ingredient = ''
    for i in range(len(infos)):
        words = jieba.cut(infos[i])
        for word in words:
            if word in keys:
                if ':' in infos[i]:
                    first_ingre = infos[i].strip().split(':')
                    ingredient += ':'.join(first_ingre[1:])

                # 解决成分有多行的情况
                k = i + 1
                while '%' in infos[k]:
                    if ':' in infos[k]:
                        if len(ingredient) != 0:
                            ingredient += ';'
                        ingredient += infos[k]
                    else:
                        ingredient += infos[k]
                    k += 1

                #解决成分标题分行的情况
                k += 1
                if k <len(infos):
                    while '%' in infos[k]:
                        if ':' in infos[k]:
                            if len(ingredient) != 0:
                                ingredient += ';'
                            ingredient += infos[k]
                        else:
                            ingredient += infos[k]
                        k += 1
                return ingredient
    return 'None'

def get_price(infos):
    """
        提取价格,价格由数字和.组成
    Args:
        infos: 文本信息
    Returns:
        price: 商品价格
    """
    for info in infos:
        if u'￥' in info:
            #正则匹配，提取出价格
            re_price = re.compile(r'\d+\.?\d+')
            price = re_price.findall(info)[0]
            return price
    return 'None'

def get_style(infos):
    """
        提取货号,货号由数字和字母组成
    Args:
        infos: 文本信息
    Return:
        style: 货号
    """
    keys = config.get('KEYS','styles').split(',')
    style = ''
    for i in range(len(infos)):
        words = jieba.cut(infos[i])
        for word in words:
            if word in keys:
                if ':' in infos[i]:
                    first_ingre = infos[i].strip().split(':')
                    re_style = re.compile(r'\w+')
                    style += re_style.findall(first_ingre[1])[0]
                    return style
    return 'None'

def get_info(url):
    """
        整合提取的信息
    Args:
        url: 图片链接
    Returns:
         key_dicts 以字典的形式返回提取的数据
    """
    image = get_file_content(url)

    key_dicts = {}

    result = client.basicAccurate(image)
    infos = []
    for item in result['words_result']:
        for info in item.values():
            infos.append(info)

    ingre = get_ingredient(infos)
    prices = get_price(infos)
    styles = get_style(infos)
    key_dicts['货号'] = styles
    key_dicts['成分'] = ingre
    key_dicts['零售价'] = prices

    return key_dicts

if __name__=='__main__':
    path = 'aptitude_top1000'
    files = os.listdir(path)
    drop_list = []
    for file in files:
        if len(drop_list)>=5:
            break
        if file[0] == '0':
            drop_list.append(file)
    with open('result1.txt','w') as rf:
        for item in drop_list:
            print(item)
            key_dicts = get_info(path + '/' + item)
            rf.write(item + '\n')
            rf.write('货号：' + key_dicts['货号'].encode('gbk') + '\n')
            rf.write('成分：' + key_dicts['成分'] + '\n')
            rf.write('零售价：' + key_dicts['零售价'] + '\n')
