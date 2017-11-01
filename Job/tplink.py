# def tplink ():
#     n = 120
#     result = []
#     while (n>=0):
#         s = 'tplink'.split()
#         while(len(s)!= 0):
#             for item in s :
#                 temp = None
#                 temp += item
#                 s = s.remove(item)
#         n -= 1
#         result.append(temp)
# while 1:
#     try:
#         s = input().split(' ')
#         print(s)
#         only = list(set(s))
#         print(only)
#         total = len(s)
#         ids = []
#         for item in only:
#             temp = 0
#             for itemp in s:
#                 if itemp == item:
#                     temp += 1
#             if 3*temp >= total:
#                 ids.append(item)
#         if ids == None:
#             print(-1)
#         else:
#             print(' '.join(ids))
#     except:
#         break
# print('W2'.lower())
# while 1:
#     try:
#         a = []
#         while (input() != 0):
#             a.append(int(input()))
#         result = []
#         for item in a:
#             temp = 0
#             for i in range(1,item+1):
#                 if i[::-1] == i:
#                     temp += 1
#             result.append(temp)
#         for it in result:
#             print(it)
#     except:
#         break
# while 1:
#     try:
#         t = int(input())
#         qing = []
#         result = []
#         for i in range(t):
#             qing.append(int(input()))
#         for item in qing:
#             temp = 0
#             while(item != 0):
#                 n = 0
#                 while (4**n <=item):
#                     n += 1
#                 item -= 4**(n-1)
#                 temp += 1
#             result.append(temp)
#         for tt in result:
#             if tt %2 == 0:
#                 print('yang')
#             else:
#                 print('niu')
#     except:
#         break
while 1:
    try:
        t = int(input())
        qing = []
        for i in range(t):
            qing.append(int(input()))
        for item in qing:
            temp = 0
            while(item != 0):
                temp += item %4
                item = int(item/4)
            print(temp)
            if temp %2 ==0:
                print('yang')
            else:
                print('niu')
    except:
        break