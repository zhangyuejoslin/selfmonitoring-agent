import numpy as np

import matplotlib
import transformers as ppb
import torch
matplotlib.use('Agg')

from matplotlib import pyplot as plt

matrix_seen = np.load("tasks/R2R-pano/quan_maxix_unseen.npy", allow_pickle=True).item()
#matrix_unseen = np.load("tasks/R2R-pano/quan_maxix_unseen.npy").item()

for key, value in matrix_seen.items():
    #1825_2
    #4345_0
    if key == "1825_2":
       # print(value['attention_weight'])
        print('key')

        fig = plt.figure(dpi=300)

        plt.matshow(value['attention_weight'][:,:11],cmap=plt.get_cmap("YlGn"))
        """
        quan2 = ['turn around', 'exit the door to your right', 'out,', 'turn left', 'walk a', 'cross the kitchen', 'turn right into the dining room and out the other side', 'out', 'turn left', 'walk to the first door on your right', 'stop once you enter the bedroom']
        inputs = " quan ".join(quan2)
        model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        tokenized_dict = bert_tokenizer.encode_plus(inputs, add_special_tokens=True, return_attention_mask=True, return_tensors='pt', pad_to_max_length=True, max_length=80)
        quan_a = np.where(tokenized_dict["input_ids"].numpy()==24110)[1]
        quan_sum = tokenized_dict['attention_mask'].sum().data
        plt.matshow(np.flip(value['attention_weight'][:,:quan_sum],axis=1),cmap=plt.get_cmap("YlGn"))
        
       # quan1 = ['go to the left', 'then', 'turn right', 'go through the big double doors', 'veer to the left', 'go to the base of the stairs', 'then', 'go up the stairs, a couple of them,', 'then', 'stop', 'wait']
        for little_quan in quan_a:
            plt.plot([little_quan-0.5,little_quan-0.5],[-0.5,9.5],"green")
        """
        plt.savefig("tasks/R2R-pano/quan_matrix_unseen.jpg",bbox_inches='tight', pad_inches=0.0, dpi=fig.dpi)

        break


    # if value['length']>10:
    #     print(key)
    #     plt.imshow(value['attention_weight'][:,:-1])
    #     plt.savefig("tasks/R2R-pano/quan_matrix_seen.jpg",bbox_inches='tight', pad_inches=0.0)
    #     break

# ['Go', 'to', 'the', 'left',  'turn', 'right', 'go', 'through', 'the', 'big', 'double', 'doors.', 'Veer', 'to', 'the', 'left',  'go', 'to', 'the', 'base', 'of', 'the', 'stairs.', 'Then', 'go', 'up', 'the', 'stairs,', 'a', 'couple', 'of', 'them,', 'then', 'stop', 'wait.']
#str1 = "Go to the left and then turn right and go through the big double doors. Veer to the left and go to the base of the stairs.  Then go up the stairs, a couple of them, and then stop and wait. "
# str1 = str1.split(" ")
# print(str1)
# print(len(str1))
