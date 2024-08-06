import pickle
import os
import json
from PIL import Image
from transformers import ViTModel, ViTImageProcessor, T5Tokenizer, T5EncoderModel
import torch
import numpy as np


def title_description_save(path_data, dataset_name):
    files = os.listdir(os.path.join(path_data, 'img'))
    list_asin = []
    for file in files:
        if '.jpg' in file:
            list_asin.append(file.split('.jpg')[0])
    asin_title_description_dict = {}
    with open(os.path.join(path_data, dataset_name+'.json'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            if 'title' in data:
                title = data['title']
            else:
                title = None
            if 'description' in data and len(data['description']) > 0:
                description = data['description'][0]
            else:
                description = None
            asin_title_description_dict[data['asin']] = [title, description]
    asin_img_tilte_description_dict = {}
    
    for asin in list_asin:
        asin_img_tilte_description_dict[asin] = asin_title_description_dict[asin]
    with open(os.path.join(path_data, 'imgs_asin_title_description.pickle'), 'wb') as f:
        pickle.dump(asin_img_tilte_description_dict, f)
    

def seq_filter_5_core(path_data, dataset_name):
    with open(os.path.join(path_data, 'imgs_asin_title_description.pickle'), 'rb') as f:
        data_asin_title_description = pickle.load(f)
    asins = list(data_asin_title_description.keys())
    
    reviewer_item_time_dict = {}
    with open(os.path.join(path_data, dataset_name+'.json'), 'r') as f:
        for data in f:
            line = json.loads(data)
            if 'asin' in line and 'reviewerID' in line and 'unixReviewTime' in line:
                asin = line['asin']
                if asin in asins:
                    reviewerid = line['reviewerID']
                    timestamp = line['unixReviewTime']
                    if reviewerid not in reviewer_item_time_dict:
                        reviewer_item_time_dict[reviewerid] = [(asin, timestamp)]
                    else:
                        reviewer_item_time_dict[reviewerid].append((asin, timestamp))
         
    filter_len_5_reviewer_item_time_dict = {}
    for reviewerid_temp in reviewer_item_time_dict:
        if len(reviewer_item_time_dict[reviewerid_temp]) >= 5:
            seq_temp, time_seq_temp = [], []
            for temp in reviewer_item_time_dict[reviewerid_temp]: 
                seq_temp.append(temp[0])
                time_seq_temp.append(temp[1])
            id_temp = sorted(range(len(time_seq_temp)), key=lambda k: time_seq_temp[k])
            filter_len_5_reviewer_item_time_dict[reviewerid_temp] = [seq_temp[i] for i in id_temp]
    
    with open(os.path.join(path_data, 'imgs_seq_5.pickle'), 'wb') as f:
        pickle.dump(filter_len_5_reviewer_item_time_dict, f)


def seq_filter_5_core_split(path_data):
    path_data_seq = os.path.join(path_data, 'imgs_seq_5.pickle')
    data_train = []
    data_val = []
    data_test = []
    with open(path_data_seq, 'rb') as f:
        data_seq = pickle.load(f)
    for user_temp in data_seq:
        data_train.append(data_seq[user_temp][:-2])
        data_val.append(data_seq[user_temp][:-1])
        data_test.append(data_seq[user_temp])
    with open(os.path.join(path_data, 'imgs_seq_5_train.pickle'), 'wb') as f:
        pickle.dump(data_train, f)
    with open(os.path.join(path_data, 'imgs_seq_5_val.pickle'), 'wb') as f:
        pickle.dump(data_val, f)
    with open(os.path.join(path_data, 'imgs_seq_5_test.pickle'), 'wb') as f:
        pickle.dump(data_test, f)


def asin2id_save(path_data):
    with open(os.path.join(path_data, 'imgs_seq_5.pickle'), 'rb') as f:
        data_seq = pickle.load(f)
    seq_list = []
    for temp in data_seq:
        seq_list += data_seq[temp]
    asin_id_dict = {}
    count = 0
    for asin_temp in seq_list:
        if asin_temp not in asin_id_dict:
            asin_id_dict[asin_temp] = count
            count += 1
    with open(os.path.join(path_data, 'asin2id.pickle'), 'wb') as f:
        pickle.dump(asin_id_dict, f)


def img_emb_vit(path_data):
    with open(os.path.join(path_data, 'asin2id.pickle'), 'rb') as f:
        data_asin_id_dict = pickle.load(f)
    image_processor = ViTImageProcessor.from_pretrained('../model_load/vit-base-patch16-224')
    vit_model = ViTModel.from_pretrained('../model_load/vit-base-patch16-224')
    img_load_path = path_data + 'img/'
    list_embs_array = []
    for img_temp in data_asin_id_dict:
        path_img = img_load_path + img_temp + '.jpg'
        img = Image.open(path_img)
        try:
            img_tensor = image_processor(img, return_tensors="pt")['pixel_values']
        except:
            img_tensor = image_processor(img.convert('RGB'), return_tensors="pt")['pixel_values']
        emb_temp = vit_model(img_tensor).last_hidden_state.squeeze(0)[0, :].detach().numpy()
        list_embs_array.append(emb_temp)
    embs_tensor = torch.from_numpy(np.stack(list_embs_array, axis=0))
    torch.save(embs_tensor, os.path.join(path_data, 'img_emb_vit_base_patch_16_224.pt'))
    

def asin_emb_save(path_load):
    with open(os.path.join(path_load, 'imgs_asin_title_description.pickle'), 'rb') as f:
        imgs_asin_title_description = pickle.load(f)
    with open(os.path.join(path_load, 'asin2id.pickle'), 'rb') as f:
        data_asin_id_dict = pickle.load(f)

    
    tokenizer = T5Tokenizer.from_pretrained('../model_load/t5/t5-small')
    tokenizer.padding_side="left"
    encoder = T5EncoderModel.from_pretrained('../model_load/t5/t5-small')
    list_embs_array = []
    for asin_temp in data_asin_id_dict:
        if asin_temp in imgs_asin_title_description:
            title = '</s>' + imgs_asin_title_description[asin_temp][0]
            title_tokenizer = tokenizer(title, return_tensors="pt", truncation=True, padding='max_length', max_length=15)
            last_hidden_state = encoder(**title_tokenizer).last_hidden_state
            rep = last_hidden_state[:,-1,:].squeeze(0)
        else:
            title = "None"
            title_tokenizer = '</s>' + tokenizer(title, return_tensors="pt", truncation=True, padding='max_length', max_length=15)
            last_hidden_state = encoder(**title_tokenizer).last_hidden_state
            rep = last_hidden_state[:,-1,:].squeeze(0)
        list_embs_array.append(rep.detach().numpy())
    embs_tensor = torch.from_numpy(np.stack(list_embs_array, axis=0))
    torch.save(embs_tensor, os.path.join(path_load, 'title_emb_t5_encoder_small_10_s.pt'))
        
        
def asin_emb_title_all_save(path_load):
    with open(os.path.join(path_load, 'imgs_asin_title_description.pickle'), 'rb') as f:
        imgs_asin_title_description = pickle.load(f)
    with open(os.path.join(path_load, 'asin2id.pickle'), 'rb') as f:
        data_asin_id_dict = pickle.load(f)
    
    tokenizer = T5Tokenizer.from_pretrained('../model_load/t5/t5-base')
    tokenizer.padding_side="left"
    encoder = T5EncoderModel.from_pretrained('../model_load/t5/t5-base')
    list_embs_array = []
    for asin_temp in data_asin_id_dict:
        if asin_temp in imgs_asin_title_description:
            title = imgs_asin_title_description[asin_temp][0]
            title_tokenizer = tokenizer(title, return_tensors="pt", truncation=True, padding='max_length', max_length=16)
            last_hidden_state = encoder(**title_tokenizer).last_hidden_state

            rep = last_hidden_state[:,-1,:].squeeze(0)
        else:
            title = "None"
            title_tokenizer = tokenizer(title, return_tensors="pt", truncation=True, padding='max_length', max_length=16)
            last_hidden_state = encoder(**title_tokenizer).last_hidden_state
            rep = last_hidden_state[:,-1,:].squeeze(0)
        list_embs_array.append(rep.detach().numpy())
    embs_tensor = torch.from_numpy(np.stack(list_embs_array, axis=0))
    torch.save(embs_tensor, os.path.join(path_load, 'title_emb_t5_encoder_small_10_s.pt'))
        
        
def main():
    path_load = 'Sports/'
    title_description_save(path_data=path_load, dataset_name='Sports')
    seq_filter_5_core(path_data=path_load, dataset_name='Sports')
    seq_filter_5_core_split(path_load)
    asin2id_save(path_load)
    img_emb_vit(path_load)
    asin_emb_save(path_load=path_load)
    


if __name__ =="__main__":
    main()
    