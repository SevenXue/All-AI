# leetcode-5 最长回文子串
# manacher算法

from time import time
def manacher(s):
    s = '#' + '#'.join(s) + '#'

    RL = [0] * len(s)
    MaxRight = 0
    pos = 0
    MaxLen = 0

    for i in range(len(s)):
        if i < MaxRight:
            RL[i] = min(RL[2*pos - i], MaxRight - i)
        else:
            RL[i] = 1
        while i - RL[i] >= 0 and i + RL[i] < len(s) and s[i - RL[i]] == s[i + RL[i]]:
            RL[i] += 1
        if RL[i] + i - 1 > MaxRight:
            MaxRight = RL[i] + i - 1
            pos = i
        # if RL[i] >= MaxLen:
        #     pos = i
        MaxLen = max(MaxLen, RL[i])

    result = s[pos - MaxLen + 1:pos + MaxLen]
    return ''.join([result[i] for i in range(1, len(result), 2)])

if __name__ == '__main__':
    s = 'babadshshshddwowbnsha'
    a = manacher(s)
    print(a)