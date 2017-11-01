# while True:
#     try:
#         n = int(input())
#         #print(n)
#         lista = [int(i) for i in input().split( )]
#         lista = sorted(lista)
#         #print(lista)
#         listb = [lista[i]-lista[i+1] for i in range(0,n-1)]
#         #print(listb)
#         if max(listb) == min(listb):
#             print('Possible')
#         else:
#             print('Impossible')
#     except:
#         break


# while True:
#     try:
#         n = int(input())
#         s = []
#         while (n!=0):
#             if n%2 == 0:
#                 s.append('2')
#                 n = (n-2)/2
#             else:
#                 s.append('1')
#                 n = (n-1)/2
#         print(s)
#         s = s[::-1]
#         print(''.join(s))
#     except:
#         break

# while True:
#     try:
#         s = list(input())
#         num = 1
#         for i in range(0,len(s)-1):
#             if s[i+1] != s[i]:
#                 num +=1
#         aver = float(len(s)/num)
#         print('%.2f' %aver)
#     except:
#         break
# while True:
#     try:
#         n = int(input())
#         a = [int(i) for i in input().split()]
#         b = sorted(a)
#         for i in range(0,n):
#             if a[i] == b[i]:
#                 continue
#             else:
#                 for k in range(n-1,-1,-1):
#                     if a[k] == b[i]:
#                         a[i],a[k] = a[k],a[i]
#                         break
#                 break
#         s = map(str,a)
#         print(' '.join(s))
#     except:
#         break

# def isPrime(n):
#     if n % 2 == 0:
#         return False
#     i = 3
#     while i * i <= n:
#         if n % i == 0:
#             return False
#         i += 2
#     return True
#
#
# def over(n):
#     s = str(n)[::-1]
#     return int(s)
#
#
# def cheng(s,num):
#     sums = s[0]
#     for i in range(1,8):
#         sums = s[i] + num*sums
#     return sums
# while 1:
#     try:
#         k = int(input())
#         m = []
#         for i in range(k):
#             m.append(list(map(int, input().split())))
#         #print(m)
#         n = int(input())
#         print(n)
#         temp = [1 for i in range(k)]
#         print(temp)
#         templs = [cheng(m[i],temp[i]) for i in range(k)]
#         print(templs)
#         result = 0
#         while(n > 0):
#             result = min(templs)
#             print(result)
#             n = n-1
#             for i in range(k):
#                 if templs[i] == result:
#                     temp[i] +=1
#                     templs[i] = cheng(m[i],temp[i])
#                     break
#         print(result)
#     except:
#         break

# while 1:
#     try:
#         a = list(map(int, input().split()))
#         print(a[0])
#         num = 2**a[0] + 2**a[1] - 2**a[2]
#         print(sum([(num>>i&1) for i in range(0, 32)]))
#     except:
#         break
#         num = pow(2,a[0]) + pow(2,a[1]) - pow(2,a[2])
#         print sum([(num>>i&1) for i in range(0, 32)])

while True:
    try:
        n = int(input())
        m = 2
        if n == 0:
            print(1)
        elif n < 3:
            print(2)
        elif n ==3 or n == 4:
            print(3)
        else:
            ss = bin(n).replace('0b','')
            temp2 = ['1' for i in range(len(ss)-1)]
            n1 = int(''.join(temp2),2) +2
            temp = int((len(ss)-1)/2)
            if temp%2 == 0:
                m = pow(2,temp+1)-1
            else:
                m = pow(2,temp+1)-1 + pow(2,temp)
            #m = pow(2,len(ss)-2)+1
            for i in range(n1, n+1, 2):
                ss = bin(i).replace('0b', '')
                if ss == ss[::-1]:
                    m += 1
            print(m)
    except:
        break
