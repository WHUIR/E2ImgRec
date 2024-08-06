import os
import json
import requests
import time
from tqdm import tqdm
import pickle


def img_download(path_data, save_path):
    info_dict = {}
    error_info_dict = {}
    error_imag_url = {}  ## asin: [title, description]
    count_image_url = 0 
    
    # list_exist_img = os.listdir(save_path)
    # asin_img_exist = [i.split('.')[0] for i in list_exist_img]
    
    with open(path_data, 'r') as f:
        lines = f.readlines()
        lines = lines[2312346:]  ## clothing
        # lines = lines[684047:]  ## sports
        for data in tqdm(lines):
            data = json.loads(data)
          
            if 'imageURLHighRes' in data and 'asin' in data:
                if len(data['imageURLHighRes']) > 0:
                    count_image_url += 1
                    img_url = data['imageURLHighRes'][0]
                    name_img = data['asin']
                    if 'title' in data:
                        title = data['title']
                    else:
                        title = None
                    if 'description' in data and len(data['description']) > 0:
                        description = data['description'][0]
                    else:
                        description = None
                    while True:
                        try:
                            img_data = requests.get(img_url, timeout=(3.05, 22)).content
                            with open(os.path.join(save_path, name_img+'.jpg'), 'wb') as handler:
                                handler.write(img_data)
                            info_dict[name_img] = [title, description]
                            break
                        except:
                            error_imag_url[name_img] = img_url
                            error_info_dict[name_img] = [title, description]
                            time.sleep(0.001)    
                
    with open(os.path.join(save_path, 'error_imgs.pickle'), 'wb') as f:
        pickle.dump(error_imag_url, f)
    with open(os.path.join(save_path.split('/')[0], '/asin_title_description.pickle'), 'wb') as f_info:
        pickle.dump(info_dict, f_info)
        

def main():
    path_data = 'Sports/meta_Sports_and_Outdoors.json'
    save_data = 'Sports/img/'
    img_download(path_data, save_data)


if __name__ =="__main__":
    main()
