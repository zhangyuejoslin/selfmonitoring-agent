import torch
import torch.nn as nn
from torch.autograd import Variable

from models.modules import build_mlp, SoftAttention, ImageSoftAttention, TextSoftAttention, PositionalEncoding, ScaledDotProductAttention, create_mask, create_mask_for_object, proj_masking, PositionalEncoding, StateAttention, ConfigAttention, ConfigObjAttention, NextSoftAttention


class SelfMonitoring1(nn.Module):
    """ An unrolled LSTM with attention over instructions for decoding navigation actions. """

    def __init__(self, opts, img_fc_dim, img_fc_use_batchnorm, img_dropout, img_feat_input_dim,
                 rnn_hidden_size, rnn_dropout, max_len, fc_bias=True, max_navigable=16):
        super(SelfMonitoring1, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_navigable = max_navigable
        self.feature_size = img_feat_input_dim
        self.hidden_size = rnn_hidden_size
        self.max_len = max_len

        proj_navigable_kwargs = {
            'input_dim': img_feat_input_dim,
            'hidden_dims': img_fc_dim,
            'use_batchnorm': img_fc_use_batchnorm,
            'dropout': img_dropout,
            'fc_bias': fc_bias,
            'relu': opts.mlp_relu
        }
        self.proj_navigable_mlp = build_mlp(**proj_navigable_kwargs)

        self.h0_fc = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=fc_bias)
        self.h1_fc = nn.Linear(rnn_hidden_size, rnn_hidden_size, bias=fc_bias)

        self.text_soft_attn = TextSoftAttention()
        self.img_soft_attn = ImageSoftAttention()
        self.soft_attn = SoftAttention()

        self.r_linear = nn.Linear(rnn_hidden_size + 128, 2)
        self.config_linear = nn.Linear(768, 512)

        self.sm = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(p=rnn_dropout)

        self.state_attention = StateAttention()

        self.lstm = nn.LSTMCell(img_fc_dim[-1] * 2 + rnn_hidden_size, rnn_hidden_size)

        self.lang_position = PositionalEncoding(rnn_hidden_size, dropout=0.1, max_len=max_len)

        self.logit_fc = nn.Linear(rnn_hidden_size * 2, img_fc_dim[-1])
        self.h2_fc_lstm = nn.Linear(rnn_hidden_size + img_fc_dim[-1], rnn_hidden_size, bias=fc_bias)

        if opts.monitor_sigmoid:
            self.critic = nn.Sequential(
                #nn.Linear(max_len + rnn_hidden_size, 1),
                nn.Linear(15 + rnn_hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            self.critic = nn.Sequential(
                #nn.Linear(max_len + rnn_hidden_size, 1),
                nn.Linear(15 + rnn_hidden_size, 1),
                nn.Tanh()
            )

        self.num_predefined_action = 1
        self.r_transform = Variable(torch.tensor([[1,0,0.75,0.5],[0,1,0.25,0.5]]).transpose(0,1), requires_grad=False)

    def forward(self, img_feat, navigable_feat, pre_feat, question, h_0, c_0, ctx, pre_ctx_attend,
               s0, r_t, navigable_index=None, ctx_mask=None):
        """ Takes a single step in the decoder

        img_feat: batch x 36 x feature_size
        navigable_feat: batch x max_navigable x feature_size

        pre_feat: previous attended feature, batch x feature_size

        question: this should be a single vector representing instruction

        ctx: batch x seq_len x dim
        navigable_index: list of list
        ctx_mask: batch x seq_len - indices to be masked
        """
        batch_size, num_imgs, feat_dim = img_feat.size()

        index_length = [len(_index) + self.num_predefined_action for _index in navigable_index]
        navigable_mask = create_mask(batch_size, self.max_navigable, index_length)

        proj_navigable_feat = proj_masking(navigable_feat, self.proj_navigable_mlp, navigable_mask)
        proj_pre_feat = self.proj_navigable_mlp(pre_feat)
        positioned_ctx = self.lang_position(ctx)
        weighted_ctx, ctx_attn = self.soft_attn(self.h1_fc(h_0), positioned_ctx, mask=ctx_mask)


        weighted_img_feat, img_attn = self.soft_attn(self.h0_fc(h_0), proj_navigable_feat, mask=navigable_mask)

        # if r_t is None:
        # r_t = self.r_linear(torch.cat((weighted_img_feat, h_0), dim=1)) 
        # r_t = self.sm(r_t)
        #change to 4 states
        # new_r_transform = self.r_transform.to(r_t.device)
        # new_r_t = torch.matmul(r_t, new_r_transform)
        

        #weighted_ctx, ctx_attn = self.state_attention(s0, r_t, ctx, ctx_mask)

        # merge info into one LSTM to be carry through time
        concat_input = torch.cat((proj_pre_feat, weighted_img_feat, weighted_ctx), 1)

        h_1, c_1 = self.lstm(concat_input, (h_0, c_0))
        h_1_drop = self.dropout(h_1)

        # policy network
        h_tilde = self.logit_fc(torch.cat((weighted_ctx, h_1_drop), dim=1))
        logit = torch.bmm(proj_navigable_feat, h_tilde.unsqueeze(2)).squeeze(2)

        # value estimation
        concat_value_input = self.h2_fc_lstm(torch.cat((h_0, weighted_img_feat), 1))

        h_1_value = self.dropout(torch.sigmoid(concat_value_input) * torch.tanh(c_1))

        value = self.critic(torch.cat((ctx_attn, h_1_value), dim=1))

        return h_1, c_1, weighted_ctx, img_attn, ctx_attn, logit, value, navigable_mask



class SelfMonitoring(nn.Module):
    """ An unrolled LSTM with attention over instructions for decoding navigation actions. """

    def __init__(self, opts, img_fc_dim, img_fc_use_batchnorm, img_dropout, img_feat_input_dim,
                 rnn_hidden_size, rnn_dropout, max_len, fc_bias=True, max_navigable=16):
        super(SelfMonitoring, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_navigable = max_navigable
        self.feature_size = img_feat_input_dim
        self.hidden_size = rnn_hidden_size
        self.max_len = max_len

        proj_navigable_kwargs = {
            'input_dim': img_feat_input_dim,
            'hidden_dims': img_fc_dim,
            'use_batchnorm': img_fc_use_batchnorm,
            'dropout': img_dropout,
            'fc_bias': fc_bias,
            'relu': opts.mlp_relu
        }
        self.proj_navigable_mlp = build_mlp(**proj_navigable_kwargs)

        self.h0_fc = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=fc_bias)
        self.h1_fc = nn.Linear(rnn_hidden_size, rnn_hidden_size, bias=fc_bias)

        self.soft_attn = SoftAttention()

        self.dropout = nn.Dropout(p=rnn_dropout)

        self.lstm = nn.LSTMCell(img_fc_dim[-1] * 2 + rnn_hidden_size, rnn_hidden_size)

        self.lang_position = PositionalEncoding(rnn_hidden_size, dropout=0.1, max_len=max_len)

        self.logit_fc = nn.Linear(rnn_hidden_size * 2, img_fc_dim[-1])
        self.h2_fc_lstm = nn.Linear(rnn_hidden_size + img_fc_dim[-1], rnn_hidden_size, bias=fc_bias)

        if opts.monitor_sigmoid:
            self.critic = nn.Sequential(
                nn.Linear(max_len + rnn_hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            self.critic = nn.Sequential(
                nn.Linear(max_len + rnn_hidden_size, 1),
                nn.Tanh()
            )

        self.num_predefined_action = 1


    def forward(self, img_feat, navigable_feat, pre_feat, question, h_0, c_0, ctx, pre_ctx_attend,
                navigable_index=None, ctx_mask=None):
        """ Takes a single step in the decoder
        img_feat: batch x 36 x feature_size
        navigable_feat: batch x max_navigable x feature_size
        pre_feat: previous attended feature, batch x feature_size
        question: this should be a single vector representing instruction
        ctx: batch x seq_len x dim
        navigable_index: list of list
        ctx_mask: batch x seq_len - indices to be masked
        """
        batch_size, num_imgs, feat_dim = img_feat.size()

        index_length = [len(_index) + self.num_predefined_action for _index in navigable_index]
        navigable_mask = create_mask(batch_size, self.max_navigable, index_length)

        proj_navigable_feat = proj_masking(navigable_feat, self.proj_navigable_mlp, navigable_mask)
        proj_pre_feat = self.proj_navigable_mlp(pre_feat)
        positioned_ctx = self.lang_position(ctx)

        weighted_ctx, ctx_attn = self.soft_attn(self.h1_fc(h_0), positioned_ctx, mask=ctx_mask)

        weighted_img_feat, img_attn = self.soft_attn(self.h0_fc(h_0), proj_navigable_feat, mask=navigable_mask)

        # merge info into one LSTM to be carry through time
        concat_input = torch.cat((proj_pre_feat, weighted_img_feat, weighted_ctx), 1)

        h_1, c_1 = self.lstm(concat_input, (h_0, c_0))
        h_1_drop = self.dropout(h_1)

        # policy network
        h_tilde = self.logit_fc(torch.cat((weighted_ctx, h_1_drop), dim=1))
        logit = torch.bmm(proj_navigable_feat, h_tilde.unsqueeze(2)).squeeze(2)

        # value estimation
        concat_value_input = self.h2_fc_lstm(torch.cat((h_0, weighted_img_feat), 1))

        h_1_value = self.dropout(torch.sigmoid(concat_value_input) * torch.tanh(c_1))

        value = self.critic(torch.cat((ctx_attn, h_1_value), dim=1))

        return h_1, c_1, weighted_ctx, img_attn, ctx_attn, logit, value, navigable_mask


class SpeakerFollowerBaseline(nn.Module):
    """ An unrolled LSTM with attention over instructions for decoding navigation actions. """

    def __init__(self, opts, img_fc_dim, img_fc_use_batchnorm, img_dropout, img_feat_input_dim,
                 rnn_hidden_size, rnn_dropout, max_len, fc_bias=True, max_navigable=16):
        super(SpeakerFollowerBaseline, self).__init__()

        self.max_navigable = max_navigable
        self.feature_size = img_feat_input_dim
        self.hidden_size = rnn_hidden_size

        self.proj_img_mlp = nn.Linear(img_feat_input_dim, img_fc_dim[-1], bias=fc_bias)

        self.proj_navigable_mlp = nn.Linear(img_feat_input_dim, img_fc_dim[-1], bias=fc_bias)

        self.h0_fc = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=False)

        self.soft_attn = SoftAttention()

        self.dropout = nn.Dropout(p=rnn_dropout)

        self.lstm = nn.LSTMCell(img_feat_input_dim * 2, rnn_hidden_size)

        self.h1_fc = nn.Linear(rnn_hidden_size, rnn_hidden_size, bias=False)

        self.proj_out = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=fc_bias)


        

    def forward(self, img_feat, navigable_feat, pre_feat, h_0, c_0, ctx, navigable_index=None, ctx_mask=None):
        """ Takes a single step in the decoder LSTM.

        img_feat: batch x 36 x feature_size
        navigable_feat: batch x max_navigable x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        navigable_index: list of list
        ctx_mask: batch x seq_len - indices to be masked
        """
        batch_size, num_imgs, feat_dim = img_feat.size()

        index_length = [len(_index) + self.num_predefined_action for _index in navigable_index]
        navigable_mask = create_mask(batch_size, self.max_navigable, index_length)

        proj_navigable_feat = proj_masking(navigable_feat, self.proj_navigable_mlp, navigable_mask)
        proj_pre_feat = self.proj_navigable_mlp(pre_feat)
        positioned_ctx = self.lang_position(ctx)

        weighted_ctx, ctx_attn = self.soft_attn(self.h1_fc(h_0), positioned_ctx, mask=ctx_mask)

        weighted_img_feat, img_attn = self.soft_attn(self.h0_fc(h_0), proj_navigable_feat, mask=navigable_mask)

        # merge info into one LSTM to be carry through time
        concat_input = torch.cat((proj_pre_feat, weighted_img_feat, weighted_ctx), 1)

        h_1, c_1 = self.lstm(concat_input, (h_0, c_0))
        h_1_drop = self.dropout(h_1)

        # policy network
        h_tilde = self.logit_fc(torch.cat((weighted_ctx, h_1_drop), dim=1))
        logit = torch.bmm(proj_navigable_feat, h_tilde.unsqueeze(2)).squeeze(2)

        # value estimation
        concat_value_input = self.h2_fc_lstm(torch.cat((h_0, weighted_img_feat), 1))

        h_1_value = self.dropout(torch.sigmoid(concat_value_input) * torch.tanh(c_1))

        value = self.critic(torch.cat((ctx_attn, h_1_value), dim=1))

        return h_1, c_1, weighted_ctx, img_attn, ctx_attn, logit, value, navigable_mask



class Configuring(nn.Module):

    def __init__(self, opts, img_fc_dim, img_fc_use_batchnorm, img_dropout, img_feat_input_dim,
                 rnn_hidden_size, rnn_dropout, max_len, fc_bias=True, max_navigable=16):
        super(Configuring, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_navigable = max_navigable
        self.feature_size = img_feat_input_dim
        self.hidden_size = rnn_hidden_size
        self.max_len = max_len

        proj_navigable_kwargs = {
            'input_dim': img_feat_input_dim,
            'hidden_dims': img_fc_dim,
            'use_batchnorm': img_fc_use_batchnorm,
            'dropout': img_dropout,
            'fc_bias': fc_bias,
            'relu': opts.mlp_relu
        }
        self.proj_navigable_mlp = build_mlp(**proj_navigable_kwargs)

        self.h0_fc = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=fc_bias)
        self.h1_fc = nn.Linear(rnn_hidden_size, rnn_hidden_size, bias=fc_bias)

        self.soft_attn = SoftAttention()

        self.dropout = nn.Dropout(p=rnn_dropout)

        self.lstm = nn.LSTMCell(img_fc_dim[-1] * 2 + rnn_hidden_size, rnn_hidden_size)

        self.lang_position = PositionalEncoding(rnn_hidden_size, dropout=0.1, max_len=max_len)

        self.logit_fc = nn.Linear(rnn_hidden_size * 2, img_fc_dim[-1])

        self.h2_fc_lstm = nn.Linear(rnn_hidden_size + img_fc_dim[-1], rnn_hidden_size, bias=fc_bias)

        self.r_linear = nn.Linear(rnn_hidden_size + 128, 2)
    
        self.sm = nn.Softmax(dim=1)

        self.num_predefined_action = 1

        self.state_attention = StateAttention()

        self.config_fc = nn.Linear(768, 512, bias=False)

        if opts.monitor_sigmoid:
            self.critic = nn.Sequential(
                #nn.Linear(max_len + rnn_hidden_size, 1),
                nn.Linear(10 + rnn_hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            self.critic = nn.Sequential(
               # nn.Linear(max_len + rnn_hidden_size, 1),
                nn.Linear(10 + rnn_hidden_size, 1),
                nn.Tanh()
            )

    def forward(self, img_feat, navigable_feat, pre_feat, question, h_0, c_0, ctx, pre_ctx_attend,
                s_0, r_t, navigable_index=None, ctx_mask=None):

    #def forward(self, img_feat, navigable_feat, pre_feat, question, h_0, c_0, ctx, pre_ctx_attend,\
               # navigable_index=None, ctx_mask=None, s_0, r_t, config_embedding):

        """ Takes a single step in the decoder LSTM.
        config_embedding: batch x max_config_len x config embeddding
        image_feature: batch x 12 images  x image_feature_size
        navigable_index: list of navigable viewstates
        h_t: batch x hidden_size
        c_t: batch x hidden_size
        ctx_mask: batch x seq_len - indices to be masked
        """
        batch_size, num_imgs, feat_dim = img_feat.size()

        index_length = [len(_index) + self.num_predefined_action for _index in navigable_index]
        navigable_mask = create_mask(batch_size, self.max_navigable, index_length)

        proj_navigable_feat = proj_masking(navigable_feat, self.proj_navigable_mlp, navigable_mask)
        proj_pre_feat = self.proj_navigable_mlp(pre_feat)

        weighted_img_feat, img_attn = self.soft_attn(self.h0_fc(h_0), proj_navigable_feat, mask=navigable_mask)

        
        if r_t is None:
            r_t = self.r_linear(torch.cat((weighted_img_feat, h_0), dim=1)) 
            r_t = self.sm(r_t)
        
        
        # r_t = self.r_linear(torch.cat((weighted_img_feat, h_0), dim=1)) 
        # r_t = self.sm(r_t)

        weighted_ctx, ctx_attn = self.state_attention(s_0, r_t, self.config_fc(ctx), ctx_mask)
        # positioned_ctx = self.lang_position(self.config_fc(ctx))

        # weighted_ctx, ctx_attn = self.soft_attn(self.h1_fc(h_0), positioned_ctx, mask=ctx_mask)

        # merge info into one LSTM to be carry through time
        concat_input = torch.cat((proj_pre_feat, weighted_img_feat, weighted_ctx), 1)

        h_1, c_1 = self.lstm(concat_input, (h_0, c_0))
        h_1_drop = self.dropout(h_1)

        # policy network
        h_tilde = self.logit_fc(torch.cat((weighted_ctx, h_1_drop), dim=1))
        logit = torch.bmm(proj_navigable_feat, h_tilde.unsqueeze(2)).squeeze(2)

        # value estimation
        concat_value_input = self.h2_fc_lstm(torch.cat((h_0, weighted_img_feat), 1))

        h_1_value = self.dropout(torch.sigmoid(concat_value_input) * torch.tanh(c_1))

        value = self.critic(torch.cat((ctx_attn, h_1_value), dim=1))
    
        return h_1, c_1, weighted_ctx, img_attn, ctx_attn, logit, value, navigable_mask



class ConfiguringObject(nn.Module):

    def __init__(self, opts, img_fc_dim, img_fc_use_batchnorm, img_dropout, img_feat_input_dim,
                 rnn_hidden_size, rnn_dropout, max_len, fc_bias=True, max_navigable=16):
        super(ConfiguringObject, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_navigable = max_navigable
        self.feature_size = img_feat_input_dim
        self.hidden_size = rnn_hidden_size

        proj_navigable_obj_kwargs = {
            'input_dim': 152, #152
            'hidden_dims': img_fc_dim,
            'use_batchnorm': img_fc_use_batchnorm,
            'dropout': img_dropout,
            'fc_bias': fc_bias,
            'relu': opts.mlp_relu
        }
        self.proj_navigable_obj_mlp = build_mlp(**proj_navigable_obj_kwargs)

        proj_navigable_img_kwargs = {
            'input_dim': img_feat_input_dim + 36,
            'hidden_dims': img_fc_dim,
            'use_batchnorm': img_fc_use_batchnorm,
            'dropout': img_dropout,
            'fc_bias': fc_bias,
            'relu': opts.mlp_relu
        }
        self.proj_navigable_img_mlp = build_mlp(**proj_navigable_img_kwargs)

        proj_navigable_img_kwargs2 = {
            'input_dim': img_feat_input_dim,
            'hidden_dims': img_fc_dim,
            'use_batchnorm': img_fc_use_batchnorm,
            'dropout': img_dropout,
            'fc_bias': fc_bias,
            'relu': opts.mlp_relu
        }
        self.proj_navigable_img_mlp2 = build_mlp(**proj_navigable_img_kwargs2)


        self.h0_fc = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=False)
        self.next_h0_fc = nn.Linear(256, 128, bias=False)

        self.soft_attn = SoftAttention()
        
        self.next_soft_attn = NextSoftAttention()
        
        self.state_attention = StateAttention()

        self.config_obj_attention = ConfigObjAttention()

        self.dropout = nn.Dropout(p=rnn_dropout)
        
        self.lstm = nn.LSTMCell(img_fc_dim[-1] * 2 + rnn_hidden_size + 300 + 300, rnn_hidden_size)


        self.h1_fc = nn.Linear(rnn_hidden_size, rnn_hidden_size, bias=False)

        self.h2_fc_lstm = nn.Linear(rnn_hidden_size + img_fc_dim[-1], rnn_hidden_size, bias=fc_bias)

        self.proj_out = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=fc_bias)

        self.state_attention = StateAttention()


        self.logit_fc = nn.Linear(rnn_hidden_size * 2 + 300 + 300, img_fc_dim[-1])

        self.r_linear = nn.Linear(rnn_hidden_size + 128, 2)

        self.image_linear = nn.Linear(img_feat_input_dim, img_fc_dim[-1])

        self.config_fc = nn.Linear(512+300+300, 128, bias=False)

        self.config_atten_linear = nn.Linear(512, 128)
        #self.config_atten_linear = nn.Linear(768, 128)

        self.sm = nn.Softmax(dim=1)

        if opts.monitor_sigmoid:
            self.critic = nn.Sequential(
                #nn.Linear(max_len + rnn_hidden_size, 1),
                nn.Linear(15 + rnn_hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            self.critic = nn.Sequential(
               # nn.Linear(max_len + rnn_hidden_size, 1),
                nn.Linear(15 + rnn_hidden_size, 1),
                nn.Tanh()
            )

        self.r_transform = Variable(torch.tensor([[1,0,0.75,0.5],[0,1,0.25,0.5]]).transpose(0,1), requires_grad=False)
        self.ho_trans = nn.Linear(768, 512)
        self.h0_next = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=False)


    def forward(self, navigable_img_feat, navigable_obj_feat, navigable_obj_img_feat, object_mask, pre_feat, question, h_0, c_0, ctx, pre_ctx_attend, \
                s_0, r_t, navigable_index, ctx_mask, step, landmark_similarity):

        """ Takes a single step in the decoder LSTM.
        config_embedding: batch x max_config_len x config embeddding
        image_feature: batch x 12 images x 36 boxes x image_feature_size
        navigable_index: list of navigable viewstates
        h_t: batch x hidden_size
        c_t: batch x hidden_size
        ctx_mask: batch x seq_len - indices to be masked
        """
        # input of image_feature should be changed
    
        
        batch_size, num_heading, num_object, object_feat_dim = navigable_obj_feat.size()

        navigable_obj_feat = navigable_obj_feat.view(batch_size, num_heading*num_object, object_feat_dim) #4 x 16*36 x 300
        navigable_obj_img_feat = navigable_obj_img_feat.view(batch_size, num_heading*num_object, 152) # 4 x 48*36 x 152

        index_length = [len(_index)+1 for _index in navigable_index]
        
        navigable_mask = create_mask(batch_size, self.max_navigable, index_length)

        navigable_obj_mask = create_mask_for_object(batch_size, self.max_navigable*3*num_object, index_length) #batch x 48*36
        
        #obj_img_feat = torch.cat([navigable_obj_feat, navigable_obj_img_feat], dim=2)
        #proj_navigable_obj_feat = proj_masking(obj_img_feat, self.proj_navigable_obj_mlp, navigable_obj_mask)
        proj_navigable_obj_feat = proj_masking(navigable_obj_img_feat, self.proj_navigable_obj_mlp, navigable_obj_mask) # batch x 48*36 x 152 -> batch x 48*36 x 128
        proj_navigable_feat = proj_masking(torch.cat([navigable_img_feat, torch.sort(landmark_similarity, dim=-1)[0]],2), self.proj_navigable_img_mlp, navigable_mask.repeat(1,3)) # batch x 48 x 128
        #proj_navigable_feat = proj_masking(torch.cat([navigable_img_feat, landmark_similarity],2), self.proj_navigable_img_mlp, navigable_mask.repeat(1,3))
        # landmark_similarity: 4 x 48 x 36
        # navigable_img_feat: 4 x 48 x 2176  
                                                                             
        proj_pre_feat = self.proj_navigable_img_mlp2(pre_feat)

        # weighted_img_feat, img_attn = self.soft_attn(self.h0_fc(h_0), self.next_h0_fc(torch.cat([next_weighted_img_feat, proj_navigable_feat], dim=2)), mask=navigable_mask)
        weighted_img_feat, img_attn = self.soft_attn(self.h0_fc(h_0), proj_navigable_feat, mask=navigable_mask.repeat(1,3))
       # weighted_obj_feat, obj_attn = self.soft_attn(self.h0_fc(h_0), proj_navigable_obj_feat, mask=navigable_obj_mask) # batch x 128

        if r_t is None:
            r_t = self.r_linear(torch.cat((weighted_img_feat, h_0), dim=1))
            r_t = self.sm(r_t)
        
       
        # new_r_transform = self.r_transform.to(r_t.device)
        # new_r_t = torch.matmul(r_t, new_r_transform)
        # r_t0 = torch.matmul(r_t, torch.tensor([1,0,0.75,0.5], device=r_t.device))
        # r_t1 = torch.matmul(r_t, torch.tensor([0,1,0.25,0.5], device=r_t.device))
        # new_r_t = torch.stack([r_t0, r_t1], dim=1)
    
        weighted_ctx, ctx_attn = self.state_attention(s_0, r_t, ctx, ctx_mask, step)

        conf_obj_feat, conf_obj_attn = self.config_obj_attention(self.config_fc(weighted_ctx), proj_navigable_obj_feat, navigable_mask, object_mask) # 4 x 16 x 128
        weighted_conf_obj_feat, conf_obj_attn = self.soft_attn(self.h0_fc(h_0), conf_obj_feat, mask=navigable_mask.repeat(1,3)) # 4 x 128
        #conf_obj_attn = conf_obj_attn[:,0:16] + conf_obj_attn[:,16:32] + conf_obj_attn[:,32:48]
        new_weighted_img_feat = torch.bmm(conf_obj_attn.unsqueeze(dim=1), self.image_linear(navigable_img_feat)).squeeze(dim=1)# batch x 128

        # obj_attn = obj_attn.view(batch_size, num_heading, num_object) # batch x 36 x 16
        # obj_attn = torch.sum(obj_attn, dim=2) # batch x 16
        # weighted_img_feat = torch.bmm(obj_attn.unsqueeze(dim=1), self.image_linear(navigable_img_feat)).squeeze(dim=1)# batch x 2176
        
        concat_input = torch.cat((proj_pre_feat, new_weighted_img_feat, weighted_ctx), 1)

        h_1, c_1 = self.lstm(concat_input, (h_0, c_0))
        h_1_drop = self.dropout(h_1)

        # policy network
        h_tilde = self.logit_fc(torch.cat((weighted_ctx, h_1_drop), dim=1))
        logit = torch.bmm(proj_navigable_feat, h_tilde.unsqueeze(2)).squeeze(2)
        logit = logit[:,0:16] + logit[:,16:32] + logit[:,32:48]


        # how to change logit here



        # value estimation
        concat_value_input = self.h2_fc_lstm(torch.cat((h_0, weighted_img_feat), 1))

        h_1_value = self.dropout(torch.sigmoid(concat_value_input) * torch.tanh(c_1))

        value = self.critic(torch.cat((ctx_attn, h_1_value), dim=1))
    
        return h_1, c_1, weighted_ctx, conf_obj_attn, ctx_attn, logit, value, navigable_mask, r_t
    
        
        

