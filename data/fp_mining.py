import numpy as np
import pickle
import os


def fp_mine(dataset):
    with open(os.path.join(dataset, 'imgs_seq_5_train.pickle'), 'rb') as f:
        data = pickle.load(f)
    with open(dataset + '/asin2id.pickle', 'rb') as f:
        id_asin_dict = pickle.load(f)
    dict_pairs = {}
    for seq in data:
        for i in range(len(seq)-1):
            if seq[i] not in dict_pairs:
                dict_pairs[seq[i]] = {seq[i+1]: 1}
            elif seq[i+1] in dict_pairs[seq[i]]:
                dict_pairs[seq[i]][seq[i+1]] += 1
            else:
                dict_pairs[seq[i]][seq[i+1]] = 1
    dict_pairs_sort = {}
    for item in dict_pairs:
        sorted_item_nn = sorted(dict_pairs[item], key=lambda x: dict_pairs[item][x], reverse=True)
        nums = sum(list(dict_pairs[item].values()))
        dict_pairs_sort[id_asin_dict[item]] = [(id_asin_dict[i], dict_pairs[item][i]/nums) for i in sorted_item_nn]
    
    with open(os.path.join(dataset, 'pairs_freq.pickle'), 'wb') as f:
        pickle.dump(dict_pairs_sort, f)


def main():
    dataset = '/data/Sports/'
    fp_mine(dataset)


if __name__ == "__main__":
    main()

