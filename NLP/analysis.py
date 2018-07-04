data_556085 = [2, 0, 2525, 69958, 0, 0]
data_3330139 = [0, 0, 0, 52755, 0, 231]



data_2967292 = [2, 8066, 7426, 207807, 0, 420]
data_3111903 = [0, 0, 0, 106654, 0, 0]
data_3226810 = [2, 0, 16716, 10715, 0, 0]
data_3151336 = [0, 0, 0, 96138, 0, 4723]
data_1139208 = []

with open('./datasets/label/2982501_label.csv', 'r') as f:
    sms = f.readlines()
    for sm in sms:
        label = sm.strip().split(',')[0]
        if label == '5':
            print sm.strip().split(',')[1]