K = 16
class beauty():
    def __init__(self, scores):
        self.scores = scores
        self.pro = 0.5

    def modify(self, status):
        self.scores = self.scores + K * (status - self.pro)

        return None

if __name__ == '__main__':
    beauty_a = beauty(400)
    beauty_b = beauty(400)



    for i in range(10):
        beauty_a.pro = 1 / (1 + 10.0 ** ((beauty_b.scores - beauty_a.scores) / 400))
        beauty_b.pro = 1 / (1 + 10.0 ** ((beauty_a.scores - beauty_b.scores) / 400))
        stauts = int(input())
        beauty_a.modify(stauts)
        beauty_b.modify(1-stauts)
    
        print(beauty_a.scores, beauty_b.scores)

        print(beauty_a.pro, beauty_b.pro)
