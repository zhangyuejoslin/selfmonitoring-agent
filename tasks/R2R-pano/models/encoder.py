import torch
import torch.nn as nn
from torch.autograd import Variable
from models.rnn import CustomRNN
import transformers as ppb



class EncoderRNN(nn.Module):
    """ Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. """

    def __init__(self, opts, vocab_size, embedding_size, hidden_size, padding_idx,
                            dropout_ratio, bidirectional=False, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.embedding_size = embedding_size
        hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.bidirectional = bidirectional
        self.rnn_kwargs = {
            'cell_class': nn.LSTMCell,
            'input_size': embedding_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'batch_first': True,
            'dropout': 0,
        }
        self.rnn = CustomRNN(**self.rnn_kwargs)

    def create_mask(self, batchsize, max_length, length):
        """Given the length create a mask given a padded tensor"""
        tensor_mask = torch.zeros(batchsize, max_length)
        for idx, row in enumerate(tensor_mask):
            row[:length[idx]] = 1
        return tensor_mask.to(self.device)

    def flip(self, x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    def forward(self, inputs, lengths):
        """
        Expects input vocab indices as (batch, seq_len). Also requires a list of lengths for dynamic batching.
        """
        embeds = self.embedding(inputs)   # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)

        embeds_mask = self.create_mask(embeds.size(0), embeds.size(1), lengths)

        if self.bidirectional:
            output_1, (ht_1, ct_1) = self.rnn(embeds, mask=embeds_mask)
            output_2, (ht_2, ct_2) = self.rnn(self.flip(embeds, 1), mask=self.flip(embeds_mask, 1))
            output = torch.cat((output_1, self.flip(output_2, 0)), 2)
            ht = torch.cat((ht_1, ht_2), 2)
            ct = torch.cat((ct_1, ct_2), 2)
        else:
            output, (ht, ct) = self.rnn(embeds, mask=embeds_mask)

        return output.transpose(0, 1), ht.squeeze(), ct.squeeze(), embeds_mask


class EncoderConfigBert(nn.Module):
    ''' Encodes navigation configuration, returning bert representation of each configuration '''
    def __init__(self, opts, vocab_size, embedding_size, hidden_size, padding_idx,
                            dropout_ratio, bidirectional=False, num_layers=1):
        super(EncoderConfigBert, self).__init__()
        model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        self.bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.bert_model = model_class.from_pretrained(pretrained_weights)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sf = SoftAttention(dimension=embedding_size)
    
    def init_state(self, batch_size, max_config_num):
        """ Initial state of model
        a_0: batch x max_config_num
        a_0: batch x 2
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        """
        a0 = Variable(torch.zeros( 
            batch_size, 
            #max_config_num, 
            10,
            device=self.bert_model.device
            ), requires_grad=False)
        a0[:,0] = 1
        r0 = Variable(torch.zeros(
            batch_size, 
            4, 
            device=self.bert_model.device
            ), requires_grad=False)
        r0[:,0] = 1
        h0 = Variable(torch.zeros(
            batch_size,
            self.hidden_size,
            device=self.bert_model.device
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            batch_size,
            self.hidden_size,
            device=self.bert_model.device
        ), requires_grad=False)
        return a0, r0, h0, c0
    
    def bert_embedding(self, inputs, sep_list):
        start = 0
        features = []
        token_features = []
        padded_masks = []
        token_padded_masks = []
        tokenized_dict = self.bert_tokenizer.batch_encode_plus(inputs, add_special_tokens=True, return_attention_mask=True, return_tensors='pt', pad_to_max_length=True)
       
        padded = tokenized_dict['input_ids'].to(self.bert_model.device)
        attention_mask = tokenized_dict['attention_mask'].to(self.bert_model.device)

        with torch.no_grad():
            last_hidden_states = self.bert_model(padded, attention_mask=attention_mask)
        
        # len(inputs) * embedding_size (274 * 768)
        temp_feature = last_hidden_states[0]
        #max_length = max(sep_list)
        max_length = 10

        for each_sep in sep_list:
            end = start + min(each_sep, max_length)
            # len(each_sep) * embedding_size (2 * 768)
            feature = temp_feature[start:end,0,:]
           
            feature = torch.zeros(max_length, temp_feature.shape[2], device=self.bert_model.device)
            token_feature = torch.zeros(max_length, temp_feature.shape[1], temp_feature.shape[2], device=self.bert_model.device)
            token_padded_mask = torch.zeros(max_length,temp_feature.shape[1], device=self.bert_model.device)

            feature[0:(end-start), :] = temp_feature[start:end,0,:]
            token_feature[0:(end-start),:,:] = temp_feature[start:end,:,:] # 10 x 13 x 768
            
            # tmp_token_padded_mask = padded[start:end]
            # token_padded_mask[0:(end-start),:]= torch.where(tmp_token_padded_mask>0, torch.full_like(tmp_token_padded_mask, 1), tmp_token_padded_mask)
            token_padded_mask[0:(end-start),:] = attention_mask[start:end]
            start += each_sep
            # max_config_num * embedding_size (3 * 768)
    
            #feature = torch.cat((feature, torch.zeros(max_length//each_sep, feature.shape[1], device=self.bert_model.device)), dim=0)
            # 1 * max_config_num (1 * 3)
            padded_mask = torch.zeros(max_length, device=self.bert_model.device)
            padded_mask[:each_sep] = 1

            features.append(feature)
            token_features.append(token_feature)
            padded_masks.append(padded_mask)
            token_padded_masks.append(token_padded_mask)
        # batch_size * max_config_num * embedding_size (100 * 3 * 768)
        features = torch.stack(features, dim=0)
        # batch_size * 1 * max_config_num (100 * 1 * 3)
        padded_masks = torch.stack(padded_masks, dim= 0)

        token_features = torch.stack(token_features, dim=0)

        token_padded_masks = torch.stack(token_padded_masks, dim=0)

        return features, padded_masks, token_features, token_padded_masks
    
    def forward(self, inputs, sep_list):
        """
        embeds: batch x max_len_config x embedding_size
        a_t: batch x max_len_config  
        """
        embeds, padded_mask, token_features, token_padded_masks = self.bert_embedding(inputs, sep_list)  # 10 x 768
        weighted_embeds, attn = self.sf(embeds, padded_mask, token_features,token_padded_masks) # 10 x 768; 10 x  x 768

        return weighted_embeds, padded_mask

class SoftAttention(nn.Module):
    """Soft-Attention without learnable parameters
    """

    def __init__(self, dimension):
        super(SoftAttention, self).__init__()
        self.softmax = nn.Softmax(dim=2)
        self.conf_linear = nn.Linear(768, 768)

    def forward(self, cls_input, cls_mask, token_input, token_mask):
        """Propagate h through the network.
        cls_input: batch x 10 x 768
        cls_mask: batch x 10
        cls_input: batch x 10 x max_token_len x 768
        token_mask: batch x 10 x 13
        """
        # Get attention
        cls_input = self.conf_linear(cls_input)
        attn = torch.matmul(cls_input.unsqueeze(dim=2), token_input.transpose(2,3)).squeeze(2)  # batch x 10 x 13

        if token_mask is not None:
            attn.data.masked_fill_((token_mask == 0).data, -float('inf')) #batch x 10 x 13

        attn = self.softmax(attn)
        new_attn = torch.where(attn != attn, torch.zeros_like(attn), attn)
        weighted_token_input = torch.matmul(new_attn.unsqueeze(dim=2), token_input).squeeze(2) # batch x 10 x 768

        return  weighted_token_input, attn