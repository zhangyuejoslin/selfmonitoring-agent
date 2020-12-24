import json
import numpy as np
import torch
import math
from itertools import chain
import spacy
from collections import defaultdict
from tqdm import tqdm
# from itertools import chain



object_class = []
base_vocab = ['<PAD>', '<START>', '<EOS>', '<UNK>']
padding_idx = base_vocab.index('<PAD>')
specific_word = ["after", "once", "until"]
config = {
    'motion_indicator_file' : 'tasks/R2R-pano/data/data/component_data/motion_indicator/motion_dict.txt',
    'split_file' : 'tasks/R2R-pano/data/data/split_dictionary.txt',
    'stop_words_file': 'tasks/R2R-pano/data/data/stop_words.txt',
    'train_file': 'tasks/R2R-pano/data/data/R2R_train.json',
    'position_file':"tasks/R2R-pano/data/data/spatial_position_dic.txt",
    'spatial_indicator_file':"tasks/R2R-pano/data/data/spatial_indicator.txt"
}
nlp = spacy.load("en_core_web_lg")

def split_oder(dictionary):
    return sorted(dictionary, key = lambda x: len(x.split()), reverse=True)

with open(config['split_file']) as f_dict:
    dictionary = f_dict.read().split('\n')
    dictionary = split_oder(dictionary)
    tmp_dict = list(filter(None,[each_phrase+"," for each_phrase in dictionary]))
    dictionary = dictionary + tmp_dict
    dictionary = [" "+each_phrase.strip()+" " for each_phrase in dictionary]

with open(config['motion_indicator_file']) as f_dict:
    motion_dict = f_dict.read().split('\n')
    motion_dict = [each_motion.strip() for each_motion in motion_dict]
    motion_dict = split_oder(motion_dict)

with open(config["stop_words_file"]) as f_stop_word:
    stopword = f_stop_word.read().split('\n')
    stopword = split_oder(stopword)


def read_file(file_path):
    with open(file_path) as f_dict:
        read_list = f_dict.read().split('\n')
        read_list = [each.strip() for each in read_list]
        read_list = split_oder(read_list)
    return read_list

position_list = read_file(config['position_file'])  
spatial_indicator_list = read_file(config['spatial_indicator_file'])

def get_vector_represent(triplets):
    triplet_array = np.zeros((5,3,300))
    triplet_mask = np.zeros(5)
    detailed_triplet_mask = np.zeros((5,3))
    for id, each_triplet in enumerate(triplets):
        assert len(each_triplet) <= 3
        if id > 4:
            break
        triplet_mask[id] = 1
        for element_id, each_element in enumerate(each_triplet):
            tmp_list = []
            if not each_element:
                triplet_array[id][element_id] = np.zeros(300)
                continue
            detailed_triplet_mask[id][element_id] = 1
            each_element_doc = nlp(each_element)
            for e_e_d in each_element_doc:
                tmp_list.append(e_e_d.vector)
            triplet_array[id][element_id] = np.mean(np.array(tmp_list), axis=0)
    
    return triplet_array, triplet_mask, detailed_triplet_mask
        




def get_landmark2(test_sentence): 
    doc = nlp(test_sentence)
    window = 5
    start_id = 0
    end_id = 0
    landmark_stopwords = ['right', 'left','front','them', 'you','end','top', 'bottom','it','middle','side']
    connect_list = ['with','by','of','on']
    specific_token_list = ['of']
   

    triplets = []
    flag = 0
    if test_sentence in motion_dict:
        landmark_triplet = []
        triplet_vector= np.zeros((5,3,300))
        triplet_mask = np.zeros(5)
        detailed_triplet_mask = np.zeros((5,3))
        return landmark_triplet, triplet_vector, triplet_mask, detailed_triplet_mask
    else:
        #check whether two landmarks could be combined together
        noun_chunks = list(doc.noun_chunks)
        noun_chunks = [each_chunk for each_chunk in noun_chunks if each_chunk.text not in landmark_stopwords]
        chunk_length = len(noun_chunks)
        if chunk_length == 1:
            triplets.append([noun_chunks[0].text])
        else:
            for e_c_id, each_chunk in enumerate(noun_chunks):  
                if e_c_id == chunk_length-1:
                    if flag:
                        break
                    else:
                        triplets.append([each_chunk.text])
                        break
                next_chunk = noun_chunks[e_c_id+1]
                start_id = each_chunk.end
                end_id = next_chunk.start
                start_phrase = doc[start_id:end_id]
                flag = 0
                while start_id < end_id:
                    if start_phrase.text in position_list or start_phrase.text in spatial_indicator_list or start_phrase.text in specific_token_list:
                        flag = 1
                        triplets.append([each_chunk.text, start_phrase.text, next_chunk.text])
                        break
                    start_id = start_id + 1
                    start_phrase = doc[start_id: end_id]
                if not flag:
                    triplets.append([each_chunk.text])

    new_triplet = []
    for each_tr in triplets:
        tmp_list = []
        ### "and" "or" cases (only one component)
        if " and " in each_tr:
            tmp_list = each_tr.split('and') 
        elif " or " in each_tr:
            tmp_list = each_tr.split('or')
        if tmp_list:
            for each_t in tmp_list:
                new_triplet.append([each_t])
        else:
            if len(each_tr) == 3 and each_tr[1] in specific_token_list:
                new_triplet.append([" ".join(each_tr)])
            else:    
                new_triplet.append(each_tr)

    triplet_vector, triplet_mask, detailed_triplet_mask = get_vector_represent(new_triplet)
            
    return new_triplet, triplet_vector, triplet_mask, detailed_triplet_mask

if __name__ == "__main__":
       
    all_example = open("/VL/space/zhan1624/R2R-EnvDrop/r2r_src/components/config_split.txt", "r").read()
    all_example = [each_example.strip().split('\n') for each_example in all_example.split('\n\n')]
    all_landmark_list = []
    max_num = 0
    sum_num = 0
    count = 0
    for each_example in tqdm(all_example):
        for config_id, each_sentence in enumerate(each_example[2:]):
            tmp_landmark_dict = {}
            landmark_triplet, triplet_vector, triplet_mask, detailed_triplet_mask = get_landmark2(each_sentence)
            tmp_landmark_dict['instr_id'] = each_example[0]
            tmp_landmark_dict['config_id'] = each_example[0]+"_"+str(config_id)
            tmp_landmark_dict['sentence'] = each_example[1]
            if landmark_triplet:
                tmp_landmark_dict['triplets'] = landmark_triplet
            tmp_landmark_dict['configuration'] = each_sentence
            tmp_landmark_dict['triplet_vector'] = triplet_vector
            tmp_landmark_dict['triplet_mask'] = triplet_mask
            tmp_landmark_dict['detailed_triplet_mask'] = detailed_triplet_mask
            all_landmark_list.append(tmp_landmark_dict)

    np.save("/VL/space/zhan1624/selfmonitoring-agent/tasks/R2R-pano/data/data/component_data/triplets/train_landmark_triplet.npy", all_landmark_list)    



    # "walk passed the sink and stove area"
    # "stop between the refrigerator and dining table"
    #"walk through the kitchen"