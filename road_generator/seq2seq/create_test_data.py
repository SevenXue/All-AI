from matplotlib import pyplot as plt

def create_test(train_url):
    with open(train_url, 'r') as td:
        datas = td.readlines()
        blocks = {}
        block_ids = {}
        for data in datas:
            info = eval(data)
            block = str(info['block'])
            id = info['id']
            if block not in blocks.keys():
                blocks[block] = 1
                block_ids[block] = [id]
            else:
                blocks[block] += 1
                block_ids[block].append(id)

        # data statistics
        num = len(blocks.keys())
        max_block = max(blocks.values())

        print(f'the num of blocks: {num}')
        print(f"the max of one block's designs: {max_block}")

    with open('datasets/plans_select.txt', 'r') as pt:
        total = 0
        datas = pt.readlines()
        for data in datas:
            if total == 20:
                break
            data = eval(data)
            data_id = data['id']
            data_block = str(data['block'])
            if data_block in blocks.keys()\
                    and data_id not in block_ids[data_block]\
                    and blocks[data_block] <= 3:
                total += 1
                with open('datasets/test_data_same_block.txt', 'a+') as tdsb:
                    tdsb.write(str(data) + '\n')

if __name__=='__main__':
    create_test('datasets/train_data_not_extraction.txt')