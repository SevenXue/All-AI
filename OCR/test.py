#-*- coding:utf-8 -*-
from aip import AipOcr
import jieba

APP_ID = '9d6873ae49944aed952d5cac254b54be'
API_KEY = '911c37173288423a9a7f07c17fcc7f0d'
SECRET_KEY = '7460dbc32f0a49c7bf68f5dcd9ec2c8a'

client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


# url = "https//www.x.com/sample.jpg"
# result = client.basicAccurate(image, options={"language_type" : "CHN_ENG"})
# for item in result['words_result']:
#     print item.values()[0]
#     dicts = jieba.cut(item.values()[0])
#     print('/'.join(dicts))

a = u'版型:合身版'
print(':' in a)
b = a.split(':')
print ':'.join(b[1:])

a = [None ,1, 2]
print(None in a)