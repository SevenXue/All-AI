while 1:
    try:
        n = int(input())
        a = list(map(int, input().split()))
        b = sorted(a)
        result = [b[-1]]
        for i in range(len(b)-2,-1,-1):
            temp = 0
            for k in range(len(b)-1,i,-1):
                if b[k]%b[i] == 0:
                    temp = 1
                    break
            if temp == 0:
                result.append(b[i])
        print(' '.join(map(str,sorted(result))))
    except:
        break




