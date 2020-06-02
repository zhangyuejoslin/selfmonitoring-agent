import torch
import torch.nn as nn

from models.modules import build_mlp, SoftAttention, PositionalEncoding, ScaledDotProductAttention, create_mask, proj_masking, PositionalEncoding, StateAttention

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

        # add 1 because the navigable index yet count in "stay" location
        # but navigable feature does include the "stay" location at [:,0,:]
        index_length = [len(_index)+1 for _index in navigable_index]
        navigable_mask = create_mask(batch_size, self.max_navigable, index_length)

        proj_img_feat = proj_masking(img_feat, self.proj_img_mlp)

        proj_navigable_feat = proj_masking(navigable_feat, self.proj_navigable_mlp, navigable_mask)

        weighted_img_feat, _ = self.soft_attn(self.h0_fc(h_0), proj_img_feat, img_feat)

        concat_input = torch.cat((pre_feat, weighted_img_feat), 1)

        h_1, c_1 = self.lstm(self.dropout(concat_input), (h_0, c_0))

        h_1_drop = self.dropout(h_1)

        # use attention on language instruction
        weighted_context, ctx_attn = self.soft_attn(self.h1_fc(h_1_drop), self.dropout(ctx), mask=ctx_mask)
        h_tilde = self.proj_out(weighted_context)

        logit = torch.bmm(proj_navigable_feat, h_tilde.unsqueeze(2)).squeeze(2)

        return h_1, c_1, ctx_attn, logit, navigable_mask



class Configuring(nn.Module):

    def __init__(self, opts, img_fc_dim, img_fc_use_batchnorm, img_dropout, img_feat_input_dim,
                 rnn_hidden_size, rnn_dropout, max_len, fc_bias=True, max_navigable=16):
        super(Configuring, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_navigable = max_navigable
        self.feature_size = img_feat_input_dim
        self.hidden_size = rnn_hidden_size

        proj_navigable_kwargs = {
            'input_dim': img_feat_input_dim,
            'hidden_dims': img_fc_dim,
            'use_batchnorm': img_fc_use_batchnorm,
            'dropout': img_dropout,
            'fc_bias': fc_bias,
            'relu': opts.mlp_relu
        }
        self.proj_navigable_mlp = build_mlp(**proj_navigable_kwargs)

        self.h0_fc = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=False)

        self.soft_attn = SoftAttention()
        
        self.state_attention = StateAttention()

        self.dropout = nn.Dropout(p=rnn_dropout)
        
        self.lstm = nn.LSTMCell(img_fc_dim[-1] + 768, rnn_hidden_size)

        self.h1_fc = nn.Linear(rnn_hidden_size, rnn_hidden_size, bias=False)

        self.proj_out = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=fc_bias)

        self.state_attention = StateAttention()

        self.logit_fc = nn.Linear(rnn_hidden_size, img_fc_dim[-1])

        self.r_linear = nn.Linear(rnn_hidden_size + 128, 2)

        self.sm = nn.Softmax(dim=1)



    def forward(self, config_embedding, padded_mask, state_attention, image_feature, h_0, c_0, s_0, r_t, navigable_index, navigable_feat):

        """ Takes a single step in the decoder LSTM.
        config_embedding: batch x max_config_len x config embeddding
        image_feature: batch x 12 images  x image_feature_size
        navigable_index: list of navigable viewstates
        h_t: batch x hidden_size
        c_t: batch x hidden_size
        ctx_mask: batch x seq_len - indices to be masked
        """
        batch_size, num_imgs, feat_dim = image_feature.size()
        index_length = [len(_index) + 1 for _index in navigable_index]
        navigable_mask = create_mask(batch_size, self.max_navigable, index_length)
        proj_navigable_feat = proj_masking(navigable_feat, self.proj_navigable_mlp, navigable_mask) # batch x 16 x 128

        weighted_img_feat, img_attn = self.soft_attn(self.h0_fc(h_0), proj_navigable_feat, mask=navigable_mask) # batch x 128

        if r_t is None:
            r_t = self.r_linear(torch.cat((weighted_img_feat, h_0), dim=1)) 
            r_t = self.sm(r_t)
        weighted_config_feature, s_1 =  self.state_attention(s_0, r_t, config_embedding, padded_mask) # batch x 768
        concat_input = torch.cat((weighted_config_feature, weighted_img_feat), dim=1)
        h_1,c_1 = self.lstm(self.dropout(concat_input), (h_0, c_0))
        h_1_drop = self.dropout(h_1) # batch x 256
        h_tilde = self.logit_fc(h_1_drop)# batch x 128
 
        logit = torch.bmm(proj_navigable_feat, h_tilde.unsqueeze(2)).squeeze(2)

        return h_1, c_1, s_1, logit, img_attn, navigable_mask



class ConfiguringObject(nn.Module):

    def __init__(self, opts, img_fc_dim, img_fc_use_batchnorm, img_dropout, img_feat_input_dim,
                 rnn_hidden_size, rnn_dropout, max_len, fc_bias=True, max_navigable=16):
        super(ConfiguringObject, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_navigable = max_navigable
        self.feature_size = img_feat_input_dim
        self.hidden_size = rnn_hidden_size

        proj_navigable_kwargs = {
            'input_dim': img_feat_input_dim,
            'hidden_dims': img_fc_dim,
            'use_batchnorm': img_fc_use_batchnorm,
            'dropout': img_dropout,
            'fc_bias': fc_bias,
            'relu': opts.mlp_relu
        }
        self.proj_navigable_mlp = build_mlp(**proj_navigable_kwargs)

        self.h0_fc = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=False)

        self.soft_attn = SoftAttention()
        
        self.state_attention = StateAttention()

        self.dropout = nn.Dropout(p=rnn_dropout)
        
        self.lstm = nn.LSTMCell(img_fc_dim[-1] + 768, rnn_hidden_size)

        self.h1_fc = nn.Linear(rnn_hidden_size, rnn_hidden_size, bias=False)

        self.proj_out = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=fc_bias)

        self.state_attention = StateAttention()

        self.logit_fc = nn.Linear(rnn_hidden_size, img_fc_dim[-1])

        self.r_linear = nn.Linear(rnn_hidden_size + 128, 2)

        self.sm = nn.Softmax(dim=1)



    def forward(self, config_embedding, padded_mask, state_attention, image_feature, h_0, c_0, s_0, r_t, navigable_index, navigable_feat):

        """ Takes a single step in the decoder LSTM.
        config_embedding: batch x max_config_len x config embeddding
        image_feature: batch x 12 images x 36 boxes x image_feature_size
        navigable_index: list of navigable viewstates
        h_t: batch x hidden_size
        c_t: batch x hidden_size
        ctx_mask: batch x seq_len - indices to be masked
        """
        # input of image_feature should be changed
        image_feature = torch.zeros(4, 12, 36, 2048)
        navigable_feat = torch.zeros(4, 16, 36, 2048)

        batch_size, num_heading, num_object, object_feat = image_feature.size()
        navigable_feat_list = []

        index_length = [len(_index)+1 for _index in navigable_index]
        for each_index in navigable_index:
            navigable_feat_list.append(image_feature[:,each_index % 12,:,:])
            
        navigable_feat = torch.stack(navigable_feat_list, dim=1)
        navigable_feat = navigable_feat.view(4, 16*36, 2048)
        navigable_mask = create_mask(batch_size, self.max_navigable, index_length)
        # soft:
        weighted_navigable_feat = SoftAttention
        # 4 X 1 x 2048

        #concat_input = torch.cat((pre_feat, weighted_img_feat), 1)
        weighted_config_feature, s_1 =  self.state_attention(s_0, r_0, config_embedding, padded_mask)
        concat_input = weighted_config_feature
        h_1,c_1 = self.lstm(self.dropout(concat_input), (h_0.squeeze(dim = 0), c_0.squeeze(dim = 0)))
        h_1_drop = self.drop(h_1)
        h_1_drop = self.linear(2816, 2048)
        # 4 x 1 x2048
        # 4 x 2048 x1

        logit = torch.bmm(navigable_feat, h_1_drop.transpose)
        # 4 x 16*36 x 1
        logit = logit.view(4, 16, 36)
        logit = logit.max(dim=2)

        return h_1_drop, c_1, s_1, logit

        
        



