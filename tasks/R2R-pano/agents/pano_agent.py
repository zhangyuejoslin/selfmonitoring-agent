import json
import random
import numpy as np
import copy
import math
import os

import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

from utils import padding_idx, pad_list_tensors
from env import load_features

class PanoBaseAgent(object):
    """ Base class for an R2R agent with panoramic view and action. """

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.object_feat_path = 'tasks/R2R-pano/152-object_feature1.npy'
        self.object_feat = self.np_to_tensor(self.object_feat_path)
        #self.features, img_spec = load_features('img_features/ResNet-152-imagenet.tsv', True)
    
    def write_results(self):
        output = []
        new_results_path ="/".join(self.results_path.split('/')[:-1])
        if not os.path.exists(new_results_path):
            os.makedirs(new_results_path)
        for k, v in self.results.items():
            output.append(
                {
                    'instr_id': k,
                    'trajectory': v['path'],
                    'distance': v['distance'],
                    #'img_attn': v['img_attn'],
                    #'config_attn': v['config_attn'],
                   # 'value': v['value'],
                    'viewpoint_idx': v['viewpoint_idx'],
                    'navigable_idx': v['navigable_idx']
                }
            )
        
        with open(self.results_path, 'w') as f:
            json.dump(output, f)
    
    def _get_distance(self, ob):
        try:
            gt = self.gt[int(ob['instr_id'].split('_')[0])]
        except:  # synthetic data only has 1 instruction per path
            gt = self.gt[int(ob['instr_id'])]
        distance = self.env.distances[ob['scan']][ob['viewpoint']][gt['path'][-1]]
        return distance

    def _select_action(self, logit, ended, target, is_prob=False, fix_action_ended=True):
        logit_cpu = logit.clone().cpu()
        target_cpu = target.clone().cpu()
        if is_prob:
            probs = logit_cpu
        else:
            probs = F.softmax(logit_cpu, 1)     

        if self.feedback == 'argmax':
            _, action = probs.max(1)  # student forcing - argmax
            action = action.detach()
        elif self.feedback == 'teacher': #teacher forcing
            action = target_cpu.detach()

        elif self.feedback == 'sample':
            # sampling an action from model
            m = D.Categorical(probs)
            action = m.sample()
        else:
            raise ValueError('Invalid feedback option: {}'.format(self.feedback))

        # set action to 0 if already ended
        if fix_action_ended:
            for i, _ended in enumerate(ended):
                if _ended:
                    action[i] = 0

        return action

    def _next_viewpoint(self, obs, viewpoints, navigable_index, action, ended):
        next_viewpoints, next_headings = [], []
        next_viewpoint_idx = []

        for i, ob in enumerate(obs):
            if action[i] >= 1:
                next_viewpoint_idx.append(navigable_index[i][action[i] - 1])  # -1 because the first one in action is 'stop'
            else:
                next_viewpoint_idx.append('STAY')
                ended[i] = True

            # use the available viewpoints and action to select next viewpoint
            next_viewpoints.append(viewpoints[i][action[i]])
            # obtain the heading associated with next viewpoints
            next_headings.append(ob['navigableLocations'][next_viewpoints[i]]['heading'])

        return next_viewpoints, next_headings, next_viewpoint_idx, ended

    def pano_navigable_feat(self, obs, ended):

        # Get the 36 image features for the panoramic view (including top, middle, bottom)
        num_feature, feature_size = obs[0]['feature'].shape

        pano_img_feat = torch.zeros(len(obs), num_feature, feature_size)
        navigable_feat = torch.zeros(len(obs), self.opts.max_navigable, feature_size)

        navigable_feat_index, target_index, viewpoints = [], [], []
        for i, ob in enumerate(obs):
            pano_img_feat[i, :] = torch.from_numpy(ob['feature'])  # pano feature: (batchsize, 36 directions, 2048)

            index_list = []
            viewpoints_tmp = []
            gt_viewpoint_id, viewpoint_idx = ob['gt_viewpoint_idx']

            for j, viewpoint_id in enumerate(ob['navigableLocations']):
                index_list.append(int(ob['navigableLocations'][viewpoint_id]['index']))
                viewpoints_tmp.append(viewpoint_id)

                if viewpoint_id == gt_viewpoint_id:
                    # if it's already ended, we label the target as <ignore>
                    if ended[i] and self.opts.use_ignore_index:
                        target_index.append(self.ignore_index)
                    else:
                        target_index.append(j)

            # we ignore the first index because it's the viewpoint index of the current location
            # not the viewpoint index for one of the navigable directions
            # we will use 0-vector to represent the image feature that leads to "stay"
            navi_index = index_list[1:]
            navigable_feat_index.append(navi_index)
            viewpoints.append(viewpoints_tmp)

            navigable_feat[i, 1:len(navi_index) + 1] = pano_img_feat[i, navi_index]

        return pano_img_feat, navigable_feat, (viewpoints, navigable_feat_index, target_index)
    

    def obj_navigable_feat(self, obs, ended):
        
        object_num = 36
        feature_size = 152
        num_feature, img_feature_size = obs[0]['feature'].shape
        
        pano_img_feat = torch.zeros(len(obs), num_feature, img_feature_size)

        navigable_obj_feat = torch.zeros(len(obs), self.opts.max_navigable, object_num, feature_size)
        navigable_img_feat = torch.zeros(len(obs), self.opts.max_navigable, img_feature_size)
     
        next_img_feat = torch.zeros(len(obs), self.opts.max_navigable, 36, 2048)


        navigable_feat_index, target_index, viewpoints = [], [], []
        for i, ob in enumerate(obs):
            pano_img_feat[i, :] = torch.from_numpy(ob['feature'])
            index_list = []
            viewpoints_tmp = []
            heading_list = []
            next_img_feat_list = []
            gt_viewpoint_id, viewpoint_idx = ob['gt_viewpoint_idx']

            for j, viewpoint_id in enumerate(ob['navigableLocations']):
                index_list.append(int(ob['navigableLocations'][viewpoint_id]['index']))
                viewpoints_tmp.append(viewpoint_id)
                heading_list.append(ob['navigableLocations'][viewpoint_id]['rel_heading'])
                #next_img_feat_list.append(torch.tensor(self.features[ob['scan']+"_"+viewpoint_id]))

                if viewpoint_id == gt_viewpoint_id:
                    # if it's already ended, we label the target as <ignore>
                    if ended[i] and self.opts.use_ignore_index:
                        target_index.append(self.ignore_index)
                    else:
                        target_index.append(j)

            # we ignore the first index because it's the viewpoint index of the current location
            # not the viewpoint index for one of the navigable directions
            # we will use 0-vector to represent the image feature that leads to "stay"
            #next_img_feat_list = torch.stack(next_img_feat_list[1:], dim=0)
            navi_index = index_list[1:]
            navigable_feat_index.append(navi_index)
            viewpoints.append(viewpoints_tmp)
            heading_list = heading_list[1:]
            navigable_img_feat[i, 1:len(navi_index) + 1] = pano_img_feat[i, navi_index]
            navigable_obj_feat[i, 1:len(navi_index) + 1] = self._faster_rcnn_feature(ob, heading_list)
           # next_img_feat[i, 1:len(navi_index)+1] = next_img_feat_list


            # tmp_img_feat = pano_img_feat[i, navi_index][:,-128:].unsqueeze(dim=1).repeat(1,36,1)
            # navigable_obj_feat[i, 1:len(navi_index) + 1] = torch.cat([self._faster_rcnn_feature(ob, heading_list), tmp_img_feat], dim=-1)

        #return navigable_img_feat, navigable_obj_feat, next_img_feat, (viewpoints, navigable_feat_index, target_index)
        return navigable_img_feat, navigable_obj_feat, (viewpoints, navigable_feat_index, target_index)
    
    
    def np_to_tensor(self, object_feat_path):
        all_obs_obj = np.load(self.object_feat_path, allow_pickle=True).item()
        for scan_key, scan_value in all_obs_obj.items():
            for state_key, state_value in scan_value.items():
                for heading_key, heading_value in state_value.items():
                    heading_value['features'] = torch.from_numpy(heading_value['features'])

        return all_obs_obj

    
    def _faster_rcnn_feature(self, ob, heading_list):
        features = []
        for heading in heading_list:
            temp = int(round((heading*180/math.pi)/30) * 30)
            if temp >=  360:
                temp = temp - 360      
            elif temp < 0 :
                temp = temp + 360
            features.append(self.object_feat[ob['scan']][ob['viewpoint']][temp*math.pi/180]['features'])
            #features.append(torch.from_numpy(all_image_feature.item()[ob['viewpoint']][temp*math.pi/180]['features']))
        features = torch.stack(features, dim=0)

        return features

    def _sort_batch(self, obs):
        """ Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). """
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        #seq_tensor = np.array([ob for ob in obs['instr_encoding']])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]  # Full length
        seq_tensor = torch.from_numpy(seq_tensor)
        return seq_tensor.long().to(self.device), list(seq_lengths)
    
    def _config_batch(self, obs):
        """ Extract the configurations and its numbers"""
        batch_configurations = []
        configuration_num = []
        for ob in obs:
            batch_configurations += ob['configurations']
            configuration_num.append(len(ob['configurations']))
        return batch_configurations, configuration_num
    
class PanoSeq2SeqAgent(PanoBaseAgent):
    """ An agent based on an LSTM seq2seq model with attention. """
    def __init__(self, opts, env, results_path, encoder, model, feedback='sample', episode_len=20):
        super(PanoSeq2SeqAgent, self).__init__(env, results_path)
        self.opts = opts
        self.encoder = encoder
        self.model = model
        self.feedback = feedback
        self.episode_len = episode_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.ignore_index = opts.max_navigable + 1  # we define (max_navigable+1) as ignore since 15(navigable) + 1(STOP)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        self.MSELoss = nn.MSELoss()
        self.MSELoss_sum = nn.MSELoss(reduction='sum')

    def get_value_loss_from_start(self, traj, predicted_value, ended, norm_value=True, threshold=5):
        """
        This loss forces the agent to estimate how good is the current state, i.e. how far away I am from the goal?
        """
        value_target = []
        for i, _traj in enumerate(traj):
            original_dist = _traj['distance'][0]
            dist = _traj['distance'][-1]
            dist_improved_from_start = (original_dist - dist) / original_dist

            value_target.append(dist_improved_from_start)

            if dist <= 3.0:  # if we are less than 3m away from the goal
                value_target[-1] = 1

            # if ended, let us set the target to be the value so that MSE loss for that sample with be 0
            # we will average the loss according to number of not 'ended', and use reduction='sum' for MSELoss
            if ended[i]:
                value_target[-1] = predicted_value[i].detach()

        value_target = torch.FloatTensor(value_target).to(self.device)

        if self.opts.mse_sum:
            return self.MSELoss_sum(predicted_value.squeeze(), value_target) / sum(1 - ended).item()
        else:
            return self.MSELoss(predicted_value.squeeze(), value_target)

    def get_value_loss_from_start_sigmoid(self, traj, predicted_value, ended, norm_value=True, threshold=5):
        """
        This loss forces the agent to estimate how good is the current state, i.e. how far away I am from the goal?
        """
        value_target = []
        for i, _traj in enumerate(traj):
            original_dist = _traj['distance'][0]
            dist = _traj['distance'][-1]
            dist_improved_from_start = (original_dist - dist) / original_dist

            dist_improved_from_start = 0 if dist_improved_from_start < 0 else dist_improved_from_start

            value_target.append(dist_improved_from_start)

            if dist < 3.0:  # if we are less than 3m away from the goal
                value_target[-1] = 1

            # if ended, let us set the target to be the value so that MSE loss for that sample with be 0
            if ended[i]:
                value_target[-1] = predicted_value[i].detach()

        value_target = torch.FloatTensor(value_target).to(self.device)

        if self.opts.mse_sum:
            return self.MSELoss_sum(predicted_value.squeeze(), value_target) / sum(1 - ended).item()
        else:
            return self.MSELoss(predicted_value.squeeze(), value_target)

    def init_traj(self, obs):
        """initialize the trajectory"""
        batch_size = len(obs)

        traj, scan_id = [], []
        for ob in obs:
            traj.append({
                'instr_id': ob['instr_id'],
                'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
                'length': 0,
                'feature': [ob['feature']],
                'img_attn': [],
                'config_attn': [],
                'value': [],
                'progress_monitor': [],
                'action_confidence': [],
                'regret': [],
                'viewpoint_idx': [],
                'navigable_idx': [],
                'gt_viewpoint_idx': ob['gt_viewpoint_idx'],
                'steps_required': [len(ob['teacher'])],
                'distance': [super(PanoSeq2SeqAgent, self)._get_distance(ob)]
            })
            scan_id.append(ob['scan'])

        self.longest_dist = [traj_tmp['distance'][0] for traj_tmp in traj]
        self.traj_length = [1] * batch_size
        self.value_loss = torch.tensor(0).float().to(self.device)

        ended = np.array([False] * batch_size)
        last_recorded = np.array([False] * batch_size)

        return traj, scan_id, ended, last_recorded

    def update_traj(self, obs, traj, img_attn, config_attn, value, next_viewpoint_idx,
                    navigable_index, ended, last_recorded, action_prob=None):
        # Save trajectory output and accumulated reward
        for i, ob in enumerate(obs):
            if not ended[i] or not last_recorded[i]:
                traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
                dist = super(PanoSeq2SeqAgent, self)._get_distance(ob)
                traj[i]['distance'].append(dist)
                traj[i]['img_attn'].append(img_attn[i].detach().cpu().numpy().tolist())
                traj[i]['config_attn'].append(config_attn[i].detach().cpu().numpy().tolist())

                if len(value[1]) > 1:
                    traj[i]['value'].append(value[i].detach().cpu().tolist())
                else:
                    traj[i]['value'].append(value[i].detach().cpu().item())

                if action_prob is not None:
                    traj[i]['action_confidence'].append(action_prob[i].detach().cpu().item())
                    traj[i]['progress_monitor'].append((action_prob[i] * ((value[i] + 1) / 2)).detach().cpu().item())

                traj[i]['viewpoint_idx'].append(next_viewpoint_idx[i])
                traj[i]['navigable_idx'].append(navigable_index[i])
                traj[i]['steps_required'].append(len(ob['new_teacher']))
                self.traj_length[i] = self.traj_length[i] + 1
                last_recorded[i] = True if ended[i] else False

        return traj, last_recorded


    def rollout_monitor(self):
        obs = np.array(self.env.reset())  # load a mini-batch
        batch_size = len(obs)

        seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs)

        ctx, h_t, c_t, ctx_mask = self.encoder(seq, seq_lengths)
        question = h_t

        if self.opts.arch == 'progress_aware_marker' or self.opts.arch == 'iclr_marker':
            pre_feat = torch.zeros(batch_size, self.opts.img_feat_input_dim + self.opts.tiled_len).to(self.device)
        else:
            pre_feat = torch.zeros(batch_size, self.opts.img_feat_input_dim).to(self.device)

        # Mean-Pooling over segments as previously attended ctx
        pre_ctx_attend = torch.zeros(batch_size, self.opts.rnn_hidden_size).to(self.device)

        # initialize the trajectory
        traj, scan_id, ended, last_recorded = self.init_traj(obs)

        loss = 0
        for step in range(self.opts.max_episode_len):

            pano_img_feat, navigable_feat, \
            viewpoints_indices = super(PanoSeq2SeqAgent, self).pano_navigable_feat(obs, ended)
            viewpoints, navigable_index, target_index = viewpoints_indices

            pano_img_feat = pano_img_feat.to(self.device)
            navigable_feat = navigable_feat.to(self.device)
            target = torch.LongTensor(target_index).to(self.device)

            # forward pass the network
            h_t, c_t, pre_ctx_attend, img_attn, ctx_attn, logit, value, navigable_mask = self.model(
                pano_img_feat, navigable_feat, pre_feat, question, h_t, c_t, ctx,
                pre_ctx_attend, navigable_index, ctx_mask)

            # set other values to -inf so that logsoftmax will not affect the final computed loss
            logit.data.masked_fill_((navigable_mask == 0).data, -float('inf'))
            current_logit_loss = self.criterion(logit, target)
            # select action based on prediction
            action = super(PanoSeq2SeqAgent, self)._select_action(logit, ended, target, fix_action_ended=self.opts.fix_action_ended)

            if not self.opts.test_submission:
                if step == 0:
                    current_loss = current_logit_loss
                else:
                    if self.opts.monitor_sigmoid:
                        current_val_loss = self.get_value_loss_from_start_sigmoid(traj, value, ended)
                    else:
                        current_val_loss = self.get_value_loss_from_start(traj, value, ended)

                    self.value_loss += current_val_loss
                    current_loss = self.opts.value_loss_weight * current_val_loss + (
                            1 - self.opts.value_loss_weight) * current_logit_loss
            else:
                current_loss = torch.zeros(1)  # during testing where we do not have ground-truth, loss is simply 0
            loss += current_loss

            next_viewpoints, next_headings, next_viewpoint_idx, ended = super(PanoSeq2SeqAgent, self)._next_viewpoint(
                obs, viewpoints, navigable_index, action, ended)

            # make a viewpoint change in the env
            obs = self.env.step(scan_id, next_viewpoints, next_headings)

            # save trajectory output and update last_recorded
            traj, last_recorded = self.update_traj(obs, traj, img_attn, ctx_attn, value, next_viewpoint_idx,
                                                   navigable_index, ended, last_recorded)

            pre_feat = navigable_feat[torch.LongTensor(range(batch_size)), action,:]

            # Early exit if all ended
            if last_recorded.all():
                break

        self.dist_from_goal = [traj_tmp['distance'][-1] for traj_tmp in traj]

        return loss, traj

    # def rollout_monitor(self):
    #     obs = np.array(self.env.reset())  # load a mini-batch
    #     batch_size = len(obs)
    #     split_index = []
    #     token_num = 0
    #     sentence = []
    #     '''
    #     seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs)
    #     ctx, h_t, c_t, ctx_mask = self.encoder(sentence, seq_lengths) 
    #     '''

  
    #         #sentence.append(ob['instructions'])
    # #     seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs)
    # #    # ctx, ctx_mask = self.encoder(sentence, seq_lengths) 
    # #     state_attention_0, r0, h0, c0 = self.encoder.init_state(batch_size, max(configuration_num_list))
    # #     h_t = h0
    # #     c_t = c0
        

    #     ''' split and then combine
    
    #     # ctx = torch.zeros(batch_size, 80, 512, device=self.device)
    #     # h_t = []
    #     # c_t = []
    #     # ctx_mask = torch.zeros(batch_size, 80, device=self.device)
    #     # all_ctx = []
    #     # all_length = []
    #     # all_length_list = []
        
    #     # for ob_id, ob in enumerate(obs):
    #     #     seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(ob)
    #     #     all_ctx.append(seq)
    #     #     all_length_list.append(seq_lengths)
    #     #     all_length += seq_lengths
    #     # all_ctx = torch.cat(all_ctx, dim=0)

    #     # ctx_encode_tensor, tmp_h_t, tmp_c_t, ctx_encode_mask = self.encoder(all_ctx, all_length)
    #     # start = 0
    #     # for ob_id, each_l in enumerate(all_length_list):
    #     #     tmp_ctx = []
    #     #     end = start + len(each_l)
    #     #     each_ctx = ctx_encode_tensor[start:end,:]
    #     #     h_t.append(tmp_h_t[start:end][-1,:])
    #     #     c_t.append(tmp_c_t[start:end][-1,:])
    #     #     for id, each_length in enumerate(each_l) :
    #     #         tmp_ctx.append(each_ctx[id,:each_length,:])
    #     #     tmp_ctx = torch.cat(tmp_ctx, dim=0)

    #     #     if tmp_ctx.shape[0] <= 80:
    #     #         ctx[ob_id,:tmp_ctx.shape[0],:] = tmp_ctx
    #     #     else:
    #     #         ctx[ob_id,:tmp_ctx.shape[0],:] = tmp_ctx[:80, :]
    #     #     ctx_mask[ob_id,:tmp_ctx.shape[0]] = 1
    #     #     start = end
        
    #     # h_t = torch.stack(h_t)
    #     # c_t = torch.stack(c_t)
    #     '''

    #     # s0 = torch.zeros(batch_size, 80).to(self.device)
    #     # r0 = torch.zeros(batch_size, 2).to(self.device)

    #     # batch_configurations, configuration_num_list = super(PanoSeq2SeqAgent, self)._config_batch(obs)
    #     # ctx, ctx_mask= self.encoder(batch_configurations, configuration_num_list)
    #     # state_attention_0, r0, h_t, c_t = self.encoder.init_state(batch_size, max(configuration_num_list))
    #     ''' combine then split'''
    #     # for ob in obs:
    #     #     tmp_split_index = list(np.where(ob['instr_encoding']==992)[0])
    #     #     if tmp_split_index[-1] > token_num:
    #     #         token_num = tmp_split_index[-1]
    #     #     split_index.append(tmp_split_index)
    #     # batch_configurations, configuration_num_list = super(PanoSeq2SeqAgent, self)._config_batch(obs)
    #     # s0, r0, h0, c0 = self.encoder.init_state(batch_size, max(configuration_num_list))
    #     # seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs)

    #     # tmp_ctx, h_t, c_t, tmp_ctx_mask = self.encoder(seq, seq_lengths)
    #     # new_ctx = torch.zeros(batch_size, 15, token_num, 512, device = tmp_ctx.device)
    #     # new_ctx_mask = torch.zeros(batch_size, 15, token_num, device = tmp_ctx.device)
    #     # new_cls = torch.zeros(batch_size, 15, 512, device = tmp_ctx.device)
    #     # new_cls_mask = torch.zeros(batch_size, 15, device = tmp_ctx.device)
 
    #     # for ob_id, each_index_list in enumerate(split_index):
    #     #     start = 0
    #     #     for list_id, each_index in enumerate(each_index_list):
    #     #         end = each_index
    #     #         new_ctx[ob_id,list_id,0:end-start,:] = tmp_ctx[ob_id,start:end,:]
    #     #         new_ctx_mask[ob_id, list_id,0:end-start] = 1
    #     #         new_cls[ob_id, list_id, :] = tmp_ctx[ob_id, each_index,:]
    #     #         new_cls_mask[ob_id, list_id] = 1
    #     #         start = end + 1
    #     # weighted_new_ctx, attn = self.encoder.sf(new_cls, new_cls_mask, new_ctx, new_ctx_mask)
    #     # ctx = weighted_new_ctx
    #     # ctx_mask = new_cls_mask    
    #     # ctx_attn = s0
        

        
    #     # bert and split
    #     seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs)
    #     for ob in obs:
    #         sentence.append(" quan ".join(ob['configurations']) + " quan")
    #     tmp_ctx, h_t, c_t, tmp_ctx_mask, split_index = self.encoder(sentence, seq_lengths)
    #     for each_split in split_index:
    #         if each_split[-1] > token_num:
    #             token_num = each_split[-1]
    #     batch_configurations, configuration_num_list = super(PanoSeq2SeqAgent, self)._config_batch(obs)
    #     s0, r0, h0, c0 = self.encoder.init_state(batch_size, max(configuration_num_list))
    #     seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs)

        
    #     new_ctx = torch.zeros(batch_size, 15, token_num, 512, device = tmp_ctx.device)
    #     new_ctx_mask = torch.zeros(batch_size, 15, token_num, device = tmp_ctx.device)
    #     new_cls = torch.zeros(batch_size, 15, 512, device = tmp_ctx.device)
    #     new_cls_mask = torch.zeros(batch_size, 15, device = tmp_ctx.device)
 
    #     for ob_id, each_index_list in enumerate(split_index):
    #         start = 0
    #         for list_id, each_index in enumerate(each_index_list):
    #             end = each_index
    #             new_ctx[ob_id,list_id,0:end-start,:] = tmp_ctx[ob_id,start:end,:]
    #             new_ctx_mask[ob_id, list_id,0:end-start] = 1
    #             new_cls[ob_id, list_id, :] = tmp_ctx[ob_id, each_index,:]
    #             new_cls_mask[ob_id, list_id] = 1
    #             start = end + 1
    #     weighted_new_ctx, attn = self.encoder.sf(new_cls, new_cls_mask, new_ctx, new_ctx_mask)
    #     ctx = weighted_new_ctx
    #     ctx_mask = new_cls_mask    
    #     ctx_attn = s0
        
      
    #     question = h_t

    #     if self.opts.arch == 'progress_aware_marker' or self.opts.arch == 'iclr_marker':
    #         pre_feat = torch.zeros(batch_size, self.opts.img_feat_input_dim + self.opts.tiled_len).to(self.device)
    #     else:
    #         pre_feat = torch.zeros(batch_size, self.opts.img_feat_input_dim).to(self.device)

    #     # Mean-Pooling over segments as previously attended ctx
    #     pre_ctx_attend = torch.zeros(batch_size, self.opts.rnn_hidden_size).to(self.device)

    #     # initialize the trajectory
    #     traj, scan_id, ended, last_recorded = self.init_traj(obs)

    #     loss = 0
    #     for step in range(self.opts.max_episode_len):

    #         pano_img_feat, navigable_feat, \
    #         viewpoints_indices = super(PanoSeq2SeqAgent, self).pano_navigable_feat(obs, ended)
    #         viewpoints, navigable_index, target_index = viewpoints_indices

    #         pano_img_feat = pano_img_feat.to(self.device)
    #         navigable_feat = navigable_feat.to(self.device)
    #         target = torch.LongTensor(target_index).to(self.device)

    #         # forward pass the network
    #         r_t = r0 if step==0 else None
    #         h_t, c_t, pre_ctx_attend, img_attn, ctx_attn, logit, value, navigable_mask = self.model(
    #             pano_img_feat, navigable_feat, pre_feat, question, h_t, c_t, ctx, pre_ctx_attend, ctx_attn, r_t, navigable_index, ctx_mask)

    #         # set other values to -inf so that logsoftmax will not affect the final computed loss
    #         logit.data.masked_fill_((navigable_mask == 0).data, -float('inf'))
    #         current_logit_loss = self.criterion(logit, target)
    #         # select action based on prediction
    #         action = super(PanoSeq2SeqAgent, self)._select_action(logit, ended, target, fix_action_ended=self.opts.fix_action_ended)

    #         if not self.opts.test_submission:
    #             if step == 0:
    #                 current_loss = current_logit_loss
    #             else:
    #                 if self.opts.monitor_sigmoid:
    #                     current_val_loss = self.get_value_loss_from_start_sigmoid(traj, value, ended)
    #                 else:
    #                     current_val_loss = self.get_value_loss_from_start(traj, value, ended)

    #                 self.value_loss += current_val_loss
    #                 current_loss = self.opts.value_loss_weight * current_val_loss + (
    #                         1 - self.opts.value_loss_weight) * current_logit_loss
    #         else:
    #             current_loss = torch.zeros(1)  # during testing where we do not have ground-truth, loss is simply 0
    #         loss += current_loss

    #         next_viewpoints, next_headings, next_viewpoint_idx, ended = super(PanoSeq2SeqAgent, self)._next_viewpoint(
    #             obs, viewpoints, navigable_index, action, ended)

    #         # make a viewpoint change in the env
    #         obs = self.env.step(scan_id, next_viewpoints, next_headings)

    #         # save trajectory output and update last_recorded
    #         traj, last_recorded = self.update_traj(obs, traj, img_attn, ctx_attn, value, next_viewpoint_idx,
    #                                                navigable_index, ended, last_recorded)

    #         pre_feat = navigable_feat[torch.LongTensor(range(batch_size)), action,:]

    #         # Early exit if all ended
    #         if last_recorded.all():
    #             break

    #     self.dist_from_goal = [traj_tmp['distance'][-1] for traj_tmp in traj]

    #     return loss, traj

    def rollout_config(self):
        obs = np.array(self.env.reset()) # load a mini-batch
        max_config_num = 0
        batch_size = len(obs)
        #check again
        batch_configurations, configuration_num_list = super(PanoSeq2SeqAgent, self)._config_batch(obs)
        ctx, ctx_mask= self.encoder(batch_configurations, configuration_num_list)
        # ctx, ctx_mask= self.encoder(batch_configurations, configuration_num_list)
        if self.opts.arch == 'progress_aware_marker' or self.opts.arch == 'iclr_marker':
            pre_feat = torch.zeros(batch_size, self.opts.img_feat_input_dim + self.opts.tiled_len).to(self.device)
        else:
            pre_feat = torch.zeros(batch_size, self.opts.img_feat_input_dim).to(self.device)
        
        pre_ctx_attend = torch.zeros(batch_size, self.opts.rnn_hidden_size).to(self.device)
            
        # initialize the trajectory
        traj, scan_id, ended, last_recorded = self.init_traj(obs)
   
        # Do a sequence rollout and calculate the loss
        loss = 0
        state_attention_0, r0, h0, c0 = self.encoder.init_state(batch_size, max(configuration_num_list))
        ctx_attn = state_attention_0
        h_t = h0
        c_t = c0
        question = h_t

        for step in range(self.opts.max_episode_len):     
            pano_img_feat, navigable_feat, \
            viewpoints_indices = super(PanoSeq2SeqAgent, self).pano_navigable_feat(obs, ended) # should change pano_img_feat and navigable_feat
            viewpoints, navigable_index, target_index = viewpoints_indices    

            pano_img_feat = pano_img_feat.to(self.device)
            navigable_feat = navigable_feat.to(self.device)
            target = torch.LongTensor(target_index).to(self.device)
             # forward
            r_t = r0 if step==0 else None
           
            h_t, c_t, pre_ctx_attend, img_attn, ctx_attn, logit, value, navigable_mask = self.model(
                pano_img_feat, navigable_feat, pre_feat, question, h_t, c_t, ctx,
                pre_ctx_attend, ctx_attn, r_t, navigable_index, ctx_mask)
            
            # set other values to -inf so that logsoftmax will not affect the final computed loss
            logit.data.masked_fill_((navigable_mask == 0).data, -float('inf'))
            current_logit_loss = self.criterion(logit, target)
            # select action based on prediction
            action = super(PanoSeq2SeqAgent, self)._select_action(logit, ended, target, fix_action_ended=self.opts.fix_action_ended)

            if not self.opts.test_submission:
                if step == 0:
                    current_loss = current_logit_loss
                else:
                    if self.opts.monitor_sigmoid:
                        current_val_loss = self.get_value_loss_from_start_sigmoid(traj, value, ended)
                    else:
                        current_val_loss = self.get_value_loss_from_start(traj, value, ended)

                    self.value_loss += current_val_loss
                    current_loss = self.opts.value_loss_weight * current_val_loss + (
                            1 - self.opts.value_loss_weight) * current_logit_loss
            else:
                current_loss = torch.zeros(1)  # during testing where we do not have ground-truth, loss is simply 0
            loss += current_loss

            next_viewpoints, next_headings, next_viewpoint_idx, ended = super(PanoSeq2SeqAgent, self)._next_viewpoint(
                obs, viewpoints, navigable_index, action, ended)

            # make a viewpoint change in the env
            obs = self.env.step(scan_id, next_viewpoints, next_headings)

            # save trajectory output and update last_recorded
            traj, last_recorded = self.update_traj(obs, traj, img_attn, ctx_attn, value, next_viewpoint_idx,
                                                   navigable_index, ended, last_recorded)

            pre_feat = navigable_feat[torch.LongTensor(range(batch_size)), action,:]

            # Early exit if all ended
            if last_recorded.all():
                break

        self.dist_from_goal = [traj_tmp['distance'][-1] for traj_tmp in traj]

        return loss, traj
        
    
    def rollout_object_config(self):
        obs = np.array(self.env.reset()) # load a mini-batch
        batch_size = len(obs)
        token_num = 0
        split_index = []
        sentence = []
        
        """ lstm combine and split
        for ob in obs:
            tmp_split_index = list(np.where(ob['instr_encoding']==992)[0])
            if tmp_split_index[-1] > token_num:
                token_num = tmp_split_index[-1]
            split_index.append(tmp_split_index)
        batch_configurations, configuration_num_list = super(PanoSeq2SeqAgent, self)._config_batch(obs)
        s0, r0, h0, c0 = self.encoder.init_state(batch_size, max(configuration_num_list))
        seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs)

        tmp_ctx, h_t, c_t, tmp_ctx_mask = self.encoder(seq, seq_lengths)
        new_ctx = torch.zeros(batch_size, 15, token_num, 512, device = tmp_ctx.device)
        new_ctx_mask = torch.zeros(batch_size, 15, token_num, device = tmp_ctx.device)
        new_cls = torch.zeros(batch_size, 15, 512, device = tmp_ctx.device)
        new_cls_mask = torch.zeros(batch_size, 15, device = tmp_ctx.device)
 
        for ob_id, each_index_list in enumerate(split_index):
            start = 0
            for list_id, each_index in enumerate(each_index_list):
                end = each_index
                new_ctx[ob_id,list_id,0:end-start,:] = tmp_ctx[ob_id,start:end,:]
                new_ctx_mask[ob_id, list_id,0:end-start] = 1
                new_cls[ob_id, list_id, :] = tmp_ctx[ob_id, each_index,:]
                new_cls_mask[ob_id, list_id] = 1
                start = end + 1
        weighted_new_ctx, attn = self.encoder.sf(new_cls, new_cls_mask, new_ctx, new_ctx_mask)
        ctx = weighted_new_ctx
        ctx_mask = new_cls_mask    
        ctx_attn = s0
        """
        seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs)
        for ob in obs:
            sentence.append(" quan ".join(ob['configurations']) + " quan")
        tmp_ctx, h_t, c_t, tmp_ctx_mask, split_index = self.encoder(sentence, seq_lengths)
        for each_split in split_index:
            if each_split[-1] > token_num:
                token_num = each_split[-1]
        batch_configurations, configuration_num_list = super(PanoSeq2SeqAgent, self)._config_batch(obs)
        s0, r0, h0, c0 = self.encoder.init_state(batch_size, max(configuration_num_list))
        seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs)

        
        new_ctx = torch.zeros(batch_size, 15, token_num, 512, device = tmp_ctx.device)
        new_ctx_mask = torch.zeros(batch_size, 15, token_num, device = tmp_ctx.device)
        new_cls = torch.zeros(batch_size, 15, 512, device = tmp_ctx.device)
        new_cls_mask = torch.zeros(batch_size, 15, device = tmp_ctx.device)
 
        for ob_id, each_index_list in enumerate(split_index):
            start = 0
            for list_id, each_index in enumerate(each_index_list):
                end = each_index
                new_ctx[ob_id,list_id,0:end-start,:] = tmp_ctx[ob_id,start:end,:]
                new_ctx_mask[ob_id, list_id,0:end-start] = 1
                new_cls[ob_id, list_id, :] = tmp_ctx[ob_id, each_index,:]
                new_cls_mask[ob_id, list_id] = 1
                start = end + 1
        weighted_new_ctx, attn = self.encoder.sf(new_cls, new_cls_mask, new_ctx, new_ctx_mask)
        ctx = weighted_new_ctx
        ctx_mask = new_cls_mask    
        ctx_attn = s0

        if self.opts.arch == 'progress_aware_marker' or self.opts.arch == 'iclr_marker':
            pre_feat = torch.zeros(batch_size, self.opts.img_feat_input_dim + self.opts.tiled_len).to(self.device)
        else:
            pre_feat = torch.zeros(batch_size, self.opts.img_feat_input_dim).to(self.device)
        
        pre_ctx_attend = torch.zeros(batch_size, self.opts.rnn_hidden_size).to(self.device)

        traj, scan_id, ended, last_recorded = self.init_traj(obs)
   
        # Do a sequence rollout and calculate the loss
        loss = 0
        question = h_t

        for step in range(self.opts.max_episode_len):     
            navigable_img_feat, navigable_obj_feat, viewpoints_indices = super(PanoSeq2SeqAgent, self).obj_navigable_feat(obs, ended) # should change pano_img_feat and navigable_feat
            viewpoints, navigable_index, target_index = viewpoints_indices  

            navigable_img_feat = navigable_img_feat.to(self.device)
            navigable_obj_feat = navigable_obj_feat.to(self.device)
           # next_img_feat = next_img_feat.to(self.device)
            target = torch.LongTensor(target_index).to(self.device)
             # forward
            r_t = r0 if step==0 else None
            
            h_t, c_t, pre_ctx_attend, img_attn, ctx_attn, logit, value, navigable_mask = self.model(navigable_img_feat, navigable_obj_feat, pre_feat, question, \
            h_t, c_t, ctx, pre_ctx_attend, ctx_attn, r_t, navigable_index, ctx_mask)

            logit.data.masked_fill_((navigable_mask == 0).data, -float('inf'))
            current_logit_loss = self.criterion(logit, target)
            # select action based on prediction
            action = super(PanoSeq2SeqAgent, self)._select_action(logit, ended, target, fix_action_ended=self.opts.fix_action_ended)

            if not self.opts.test_submission:
                if step == 0:
                    current_loss = current_logit_loss
                else:
                    if self.opts.monitor_sigmoid:
                        current_val_loss = self.get_value_loss_from_start_sigmoid(traj, value, ended)
                    else:
                        current_val_loss = self.get_value_loss_from_start(traj, value, ended)

                    self.value_loss += current_val_loss
                    current_loss = self.opts.value_loss_weight * current_val_loss + (
                            1 - self.opts.value_loss_weight) * current_logit_loss
            else:
                current_loss = torch.zeros(1)  # during testing where we do not have ground-truth, loss is simply 0
            loss += current_loss

            next_viewpoints, next_headings, next_viewpoint_idx, ended = super(PanoSeq2SeqAgent, self)._next_viewpoint(
                obs, viewpoints, navigable_index, action, ended)

            # make a viewpoint change in the env
            obs = self.env.step(scan_id, next_viewpoints, next_headings)

            # save trajectory output and update last_recorded
            traj, last_recorded = self.update_traj(obs, traj, img_attn, ctx_attn, value, next_viewpoint_idx,
                                                   navigable_index, ended, last_recorded)

            pre_feat = navigable_img_feat[torch.LongTensor(range(batch_size)), action,:]

            # Early exit if all ended
            if last_recorded.all():
                break

        self.dist_from_goal = [traj_tmp['distance'][-1] for traj_tmp in traj]

        return loss, traj

    def rollout(self):
        obs = np.array(self.env.reset())  # load a mini-batch
        batch_size = len(obs)

        seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs)
        ctx, h_t, c_t, ctx_mask = self.encoder(seq, seq_lengths)

        pre_feat = torch.zeros(batch_size, obs[0]['feature'].shape[1]).to(self.device)

        # initialize the trajectory
        traj, scan_id = [], []
        for ob in obs:
            traj.append({
                'instr_id': ob['instr_id'],
                'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
                'length': 0,
                'feature': [ob['feature']],
                'ctx_attn': [],
                'value': [],
                'viewpoint_idx': [],
                'navigable_idx': [],
                'gt_viewpoint_idx': ob['gt_viewpoint_idx'],
                'steps_required': [len(ob['teacher'])],
                'distance': [super(PanoSeq2SeqAgent, self)._get_distance(ob)]
            })
            scan_id.append(ob['scan'])

        self.longest_dist = [traj_tmp['distance'][0] for traj_tmp in traj]
        self.traj_length = [1] * batch_size
        ended = np.array([False] * len(obs))
        last_recorded = np.array([False] * len(obs))
        loss = 0
        for step in range(self.opts.max_episode_len):

            pano_img_feat, navigable_feat, \
            viewpoints_indices = super(PanoSeq2SeqAgent, self).pano_navigable_feat(obs, ended)
            viewpoints, navigable_index, target_index = viewpoints_indices
            pano_img_feat = pano_img_feat.to(self.device)
            navigable_feat = navigable_feat.to(self.device)

            # get target
            target = torch.LongTensor(target_index).to(self.device)

            # forward pass the network
            h_t, c_t, ctx_attn, logit, navigable_mask = self.model(pano_img_feat, navigable_feat, pre_feat, h_t, c_t, ctx, navigable_index, ctx_mask)

            # we mask out output
            logit.data.masked_fill_((navigable_mask == 0).data, -float('inf'))

            loss += self.criterion(logit, target)

            # select action based on prediction
            action = super(PanoSeq2SeqAgent, self)._select_action(logit, ended)
            next_viewpoints, next_headings, next_viewpoint_idx, ended = super(PanoSeq2SeqAgent, self)._next_viewpoint(
                obs, viewpoints, navigable_index, action, ended)

            # make a viewpoint change in the env
            obs = self.env.step(scan_id, next_viewpoints, next_headings)

            # Save trajectory output and update last_recorded
            for i, ob in enumerate(obs):
                if not ended[i] or not last_recorded[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
                    dist = super(PanoSeq2SeqAgent, self)._get_distance(ob)
                    traj[i]['distance'].append(dist)
                    traj[i]['ctx_attn'].append(ctx_attn[i].detach().cpu().numpy().tolist())
                    traj[i]['viewpoint_idx'].append(next_viewpoint_idx[i])
                    traj[i]['navigable_idx'].append(navigable_index[i])
                    traj[i]['steps_required'].append(len(ob['new_teacher']))
                    self.traj_length[i] = self.traj_length[i] + 1
                    last_recorded[i] = True if ended[i] else False

            pre_feat = navigable_feat[torch.LongTensor(range(batch_size)), action,:]

            # Early exit if all ended
            if last_recorded.all():
                break

        self.dist_from_goal = [traj_tmp['distance'][-1] for traj_tmp in traj]

        return loss, traj

    def model_seen_step(self, tmp_obs, ended, tmp_pre_feat, tmp_h_t, tmp_c_t, tmp_ctx, tmp_ctx_mask,
                        tmp_question, tmp_pre_ctx_attend, tmp_ctx_attn, tmp_r_t):

        pano_img_feat, navigable_feat, viewpoints_indices = super(PanoSeq2SeqAgent, self).pano_navigable_feat(tmp_obs,
                                                                                                               ended)
        #pano_img_feat, navigable_feat, viewpoints_indices = super(PanoSeq2SeqAgent, self). obj_navigable_feat(tmp_obs,ended)
                                                                                                             
        viewpoints, navigable_index, target_index = viewpoints_indices

        pano_img_feat = pano_img_feat.to(self.device)
        navigable_feat = navigable_feat.to(self.device)

        # forward pass the network
        if self.opts.use_configuration:
            tmp_h_t, tmp_c_t, tmp_pre_ctx_attend, tmp_img_attn, tmp_ctx_attn, logit, value, navigable_mask = self.model(
                    pano_img_feat, navigable_feat, tmp_pre_feat, tmp_question, tmp_h_t, tmp_c_t, tmp_ctx,
                    tmp_pre_ctx_attend, tmp_ctx_attn, tmp_r_t, navigable_index=navigable_index, ctx_mask=tmp_ctx_mask)
        else:
            tmp_tensor = torch.zeros(5,1)
            tmp_h_t, tmp_c_t, pre_ctx_attend, img_attn, ctx_attn, logit, value, \
            navigable_mask = self.model(pano_img_feat, navigable_feat, tmp_pre_feat, tmp_question, tmp_h_t,
                                        tmp_c_t, tmp_ctx, tmp_pre_ctx_attend, tmp_tensor, tmp_tensor, navigable_index=navigable_index,
                                        ctx_mask=tmp_ctx_mask)

        # we mask out output
        logit.data.masked_fill_((navigable_mask == 0).data, -float('inf'))

        if self.opts.use_configuration:

            return viewpoints, navigable_index, navigable_feat, tmp_h_t, tmp_c_t, logit, value, tmp_img_attn, tmp_ctx_attn, tmp_pre_ctx_attend
        else:
            return viewpoints, navigable_index, navigable_feat, tmp_h_t, tmp_c_t, logit, value, img_attn, ctx_attn, pre_ctx_attend

    def sample_progress_inference(self, beam_size=5):
        """
        The constrained traditional beam search is set to mimic the "regret" action, i.e. at the viewpoint, sequentially
        choose the navigable direction according to action probability. If the progress monitor output increases,
        the agent will deterministically move forward.
        This is different from the traditional beam search, because:
        1. beams are not constrained to be from the same viewpoint
        2. all beams are completed
        :return:
        """
        obs = np.array(self.env.reset())  # load a mini-batch
        batch_size = len(obs)
        # if self.opts.use_configuration:
        batch_configurations, configuration_num_list = super(PanoSeq2SeqAgent, self)._config_batch(obs)
        ctx, ctx_mask= self.encoder(batch_configurations, configuration_num_list)
        state_attention_0, r0, h0, c0 = self.encoder.init_state(batch_size, max(configuration_num_list))
        ctx_attn = state_attention_0
        h_t = h0
        c_t = c0
        r_t = r0
           
        # else: 
        
        # seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs)
        # ctx, h_t, c_t, ctx_mask = self.encoder(seq, seq_lengths)
        tmp_tensor = torch.zeros(5,1)
        question = h_t

        self.done_beams = [[] for _ in range(batch_size)]

        batch_traj = []

        for k in range(batch_size):
            # save the visited state to avoid the agent search to a visited state
            self.visited_state = []  # we probably won't be using this during viewpoint constrained beam search

            # save all visited states for test server submission
            self.all_visited_traj = []

            # take the sample in mini-batch and duplicate for maximum number of navigable directions
            tmp_obs = np.repeat(obs[k:k + 1], beam_size)
            tmp_ctx = ctx[k:k + 1].expand(beam_size, ctx.size(1), ctx.size(2))
            tmp_h_t = h_t[k:k + 1].expand(beam_size, h_t.size(1))
            tmp_c_t = c_t[k:k + 1].expand(beam_size, c_t.size(1))

            if self.opts.use_configuration:
                tmp_ctx_attn = ctx_attn[k:k + 1].expand(beam_size, ctx_attn.size(1))
                tmp_r_t = r_t[k:k + 1].expand(beam_size, r_t.size(1))
           
            tmp_ctx_mask = ctx_mask[k:k + 1].expand(beam_size, ctx_mask.size(1))
            tmp_pre_feat = torch.zeros(beam_size, obs[0]['feature'].shape[1]).to(self.device)
            tmp_question = h_t[k:k + 1].expand(beam_size, h_t.size(1))
            tmp_pre_ctx_attend = torch.zeros(beam_size, self.opts.rnn_hidden_size).to(self.device)
            tmp_seen_feat = torch.zeros(beam_size, 1, self.opts.rnn_hidden_size).to(self.device)

            # initialize the trajectory
            traj, scan_id = [], []
            for ob in tmp_obs:
                traj.append({
                    'instr_id': ob['instr_id'],
                    'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
                    'length': 1,
                    'feature': [ob['feature']],
                    'img_attn': [],
                    'ctx_attn': [],
                    'rollback_forward_attn': [],
                    'value': [],
                    'viewpoint_idx': [],
                    'navigable_idx': [],
                    'gt_viewpoint_idx': ob['gt_viewpoint_idx'],
                    'distance': [super(PanoSeq2SeqAgent, self)._get_distance(ob)]
                })
                scan_id.append(ob['scan'])
                if (ob['viewpoint'], ob['heading']) not in self.visited_state:
                    self.visited_state.append((ob['viewpoint'], ob['heading']))

            tmp_ended = np.array([False] * beam_size)
            tmp_last_recorded = np.array([False] * beam_size)

            # agent performs the first forward step
            pano_img_feat, navigable_feat, viewpoints_indices = super(PanoSeq2SeqAgent, self).pano_navigable_feat(
                tmp_obs, tmp_ended)
            viewpoints, navigable_index, target_index = viewpoints_indices
            pano_img_feat = pano_img_feat.to(self.device)
            navigable_feat = navigable_feat.to(self.device)

            # forward pass the network
            if self.opts.use_configuration:
                tmp_h_t, tmp_c_t, tmp_pre_ctx_attend, tmp_img_attn, tmp_ctx_attn, logit, value, navigable_mask = self.model(
                    pano_img_feat, navigable_feat, tmp_pre_feat, tmp_question, tmp_h_t, tmp_c_t, tmp_ctx,
                    tmp_pre_ctx_attend, tmp_ctx_attn, tmp_r_t, navigable_index=navigable_index, ctx_mask=tmp_ctx_mask)
            else:         
                tmp_h_t, tmp_c_t, tmp_pre_ctx_attend, tmp_img_attn, tmp_ctx_attn, logit, value, navigable_mask = self.model(
                    pano_img_feat, navigable_feat, tmp_pre_feat, tmp_question, tmp_h_t, tmp_c_t, tmp_ctx,
                    tmp_pre_ctx_attend, tmp_tensor, tmp_tensor, navigable_index=navigable_index, ctx_mask=tmp_ctx_mask)

            logit.data.masked_fill_((navigable_mask == 0).data, -float('inf'))

            logprobs = F.log_softmax(logit, dim=1)
            log_values = ((value + 1) / 2).log()

            if not self.opts.use_configuration:
                tmp_r_t = tmp_tensor

            self.done_beams[k] = self.progress_inference(k, logprobs, tmp_obs, traj, tmp_ended,
                                                                            tmp_last_recorded,
                                                                            viewpoints, navigable_index, navigable_feat,
                                                                            tmp_pre_feat,
                                                                            tmp_ctx, tmp_img_attn, tmp_ctx_attn, tmp_h_t,
                                                                            tmp_c_t, tmp_ctx_mask, beam_size,
                                                                            tmp_seen_feat, tmp_question, tmp_pre_ctx_attend, tmp_r_t,
                                                                            log_values)
     

            # if self.opts.merge_traj_submission:
            assert self.done_beams[k]['traj']['path'][-1] == self.all_visited_traj[-1], 'Recorded all visited viewpoint does not end up with the same viewpoint as before'
            self.done_beams[k]['traj']['path'] = self.all_visited_traj

            batch_traj.append(self.done_beams[k]['traj'])
        return batch_traj

    def progress_inference(self, batch_idx, logprobs, tmp_obs, traj, ended, last_recorded, viewpoints,
                                          navigable_index, navigable_feat, tmp_pre_feat, tmp_ctx, tmp_img_attn,
                                          tmp_ctx_attn, tmp_h_t, tmp_c_t, tmp_ctx_mask, beam_size, tmp_seen_feat,
                                          tmp_question, tmp_pre_ctx_attend, tmp_r_t, log_values=None):

        beam_seq = torch.zeros(self.opts.max_episode_len, beam_size).to(self.device)
        beam_seq_logprobs = torch.zeros(self.opts.max_episode_len, beam_size).to(self.device)
        beam_logprobs_sum = torch.zeros(beam_size).to(self.device)  # running sum of logprobs for each beam

        scan_id = [_['scan'] for _ in tmp_obs]

        last_max_pm = -1

        for step in range(self.opts.max_episode_len):
            logprobsf = logprobs.data
            if log_values is not None:
                log_valuesf = log_values.data
            

            state = tmp_obs, traj, ended, last_recorded, viewpoints, navigable_index, navigable_feat, \
                    tmp_pre_feat, tmp_h_t, tmp_c_t, tmp_ctx, tmp_img_attn, tmp_ctx_attn, tmp_ctx_mask, \
                    tmp_seen_feat, tmp_question, tmp_pre_ctx_attend, tmp_r_t

            beam_seq, \
            beam_seq_logprobs, \
            beam_logprobs_sum, \
            state, \
            candidates_divm = self.progress_inference_step(self.visited_state, batch_idx, scan_id, logprobsf, beam_size,
                                                                   step, beam_seq, beam_seq_logprobs,
                                                                   beam_logprobs_sum, state,
                                                                   log_valuesf)

            new_obs, new_traj, new_ended, new_last_recorded, new_pre_feat, new_h_t, new_c_t, new_ctx, new_img_attn, \
            new_ctx_attn, new_ctx_mask, new_seen_feat, new_question, new_pre_ctx_attend, new_r_t= state

            # agent performs the next forward step to get value
            pano_img_feat, navigable_feat, viewpoints_indices = super(PanoSeq2SeqAgent, self).pano_navigable_feat(
                new_obs, new_ended)
            viewpoints, navigable_index, target_index = viewpoints_indices
            pano_img_feat = pano_img_feat.to(self.device)
            navigable_feat = navigable_feat.to(self.device)

            # forward pass the network
            if self.opts.use_configuration:
                new_h_t, new_c_t, new_pre_ctx_attend, new_img_attn, new_ctx_attn, logit, value, navigable_mask = self.model(
                    pano_img_feat, navigable_feat, new_pre_feat, new_question, new_h_t, new_c_t, new_ctx,
                    new_pre_ctx_attend, new_ctx_attn, new_r_t, navigable_index=navigable_index, ctx_mask=new_ctx_mask)
            else:
                 new_h_t, new_c_t, new_pre_ctx_attend, new_img_attn, new_ctx_attn, logit, value, navigable_mask = self.model(
                    pano_img_feat, navigable_feat, new_pre_feat, new_question, new_h_t, new_c_t, new_ctx,
                    new_pre_ctx_attend, new_r_t, new_r_t, navigable_index=navigable_index, ctx_mask=new_ctx_mask)

            logit.data.masked_fill_((navigable_mask == 0).data, -float('inf'))

            new_logprobs = F.log_softmax(logit, dim=1)
            new_log_values = ((value + 1) / 2).log()

            # =====================================================================
            # find the viewpoint with highest progress monitor output, and only keep the beams from that viewpoint
            progress_monitor_output = new_log_values.exp()

            # the progress monitor output is ordered according to the action probability
            # we will sequentially find the progress monitor output that is higher than previous one
            # if higher, the agent has to choose this navigable direction, even if the other navigable direction might
            # have higher progress monitor output.
            need_rollback = True
            for pm_i, pm_output in enumerate(progress_monitor_output):
                if pm_output > last_max_pm:
                    need_rollback = False
                    last_max_pm = pm_output
                    break

            if not need_rollback:
                row_max = pm_i

                # record the trajectory for test server submission
                # if this is the very first time we save traversed trajectory
                current_path = new_traj[row_max]['path']
                if self.all_visited_traj == []:
                    self.all_visited_traj = current_path
                else:
                    assert self.all_visited_traj[-1] == current_path[-2], 'the previous viewpoint does not match with recorded'
                    self.all_visited_traj.append(current_path[-1])

            else:
                last_max_pm, row_max = progress_monitor_output.max(0)

                # record the trajectory for test server submission
                for record_i in range(row_max):

                    current_path = new_traj[record_i]['path']

                    assert self.all_visited_traj[-1] == current_path[
                        -2], 'the previous viewpoint does not match with recorded'

                    # there will be duplicated trajectories because we use mini-batch to compute
                    # if the trajectory, which we are trying to record now, is the same as what was recorded
                    # we will skip it.
                    if not new_ended[record_i]:
                        if current_path[-2:] != self.all_visited_traj[-2:] and current_path[-2:][::-1] != self.all_visited_traj[-2:]:
                            self.all_visited_traj.append(current_path[-1])
                            self.all_visited_traj.append(current_path[-2])

                    elif self.all_visited_traj[-2] != self.all_visited_traj[-1]:  # if it's ended, record the last ended action, also avoid duplicated trajectories
                        self.all_visited_traj.append(current_path[-1])

                    if len(self.all_visited_traj) > 2:
                        if self.all_visited_traj[-3] == self.all_visited_traj[-2]:
                            print('what is up?')

                current_path = new_traj[row_max]['path']
                assert self.all_visited_traj[-1] == current_path[
                    -2], 'the previous viewpoint does not match with recorded'

                self.all_visited_traj.append(current_path[-1])

            # we will then keep the beams from this viewpoint selection
            # take the sample in mini-batch and duplicate for maximum number of navigable directions
            tmp_obs = np.repeat(new_obs[row_max:row_max + 1], beam_size)
            traj = np.repeat(new_traj[row_max:row_max + 1], beam_size)
            ended = np.repeat(new_ended[row_max:row_max + 1], beam_size)
            last_recorded = np.repeat(new_last_recorded[row_max:row_max + 1], beam_size)
            viewpoints = viewpoints[row_max:row_max + 1] * beam_size
            navigable_index = navigable_index[row_max:row_max + 1] * beam_size
            navigable_feat = navigable_feat[row_max:row_max + 1].repeat(beam_size, 1, 1)
            tmp_pre_feat = new_pre_feat[row_max:row_max + 1].repeat(beam_size, 1)
            tmp_h_t = new_h_t[row_max:row_max + 1].repeat(beam_size, 1)
            tmp_c_t = new_c_t[row_max:row_max + 1].repeat(beam_size, 1)
            tmp_ctx = new_ctx[row_max:row_max + 1].repeat(beam_size, 1, 1)
            tmp_img_attn = new_img_attn[row_max:row_max + 1].repeat(beam_size, 1)
            tmp_ctx_attn = new_ctx_attn[row_max:row_max + 1].repeat(beam_size, 1)
            tmp_r_t = new_r_t[row_max:row_max + 1].repeat(beam_size, 1)
            tmp_ctx_mask = new_ctx_mask[row_max:row_max + 1].repeat(beam_size, 1)
            tmp_seen_feat = new_seen_feat[row_max:row_max + 1].repeat(beam_size, 1, 1)
            tmp_question = new_question[row_max:row_max + 1].repeat(beam_size, 1)
            tmp_pre_ctx_attend = new_pre_ctx_attend[row_max:row_max + 1].repeat(beam_size, 1)
            logprobs = new_logprobs[row_max:row_max + 1].repeat(beam_size, 1)
            log_values = new_log_values[row_max:row_max + 1].repeat(beam_size, 1)
            # =====================================================================

            # if the agent stops or if time is up, copy beams
            if last_recorded.all() or step == self.opts.max_episode_len - 1:
                final_beam = {
                    'traj': traj[0]
                }
                break

        return final_beam

    def progress_inference_step(self, visited_state, batch_idx, scan_id, logprobsf, beam_size, step, beam_seq,
                                     beam_seq_logprobs, beam_logprobs_sum, state, log_valuesf=None):

        ys, ix = torch.sort(logprobsf, 1, True)

        candidates = []
        cols = min(beam_size, ys.size(1))
        rows = beam_size

        # if this is the first step, we will just use the 1st row
        # instead of use the number of beams
        if step == 0:
            rows = 1

        for c in range(cols):  # for each column (word, essentially)
            for q in range(rows):  # for each beam expansion
                # compute logprob of expanding beam q with word in (sorted) position c
                local_logprob = ys[q, c]
                candidate_logprob = beam_logprobs_sum[q] + local_logprob
                candidates.append({'c': ix[q, c], 'q': q, 'p': candidate_logprob, 'r': local_logprob})
        candidates = sorted(candidates, key=lambda x: -x['p'])

        # let us duplicate the state and then we will more stuffs around
        tmp_obs, traj, ended, last_recorded, viewpoints, navigable_index, navigable_feat, \
        tmp_pre_feat, tmp_h_t, tmp_c_t, tmp_ctx, tmp_img_attn, tmp_ctx_attn, tmp_ctx_mask, \
        tmp_seen_feat, tmp_question, tmp_pre_ctx_attend, tmp_r_t= state

        new_traj = copy.deepcopy(traj)
        new_ended = ended.copy()
        new_last_recorded = last_recorded.copy()
        new_pre_feat = tmp_pre_feat.clone()
        new_tmp_h_t = tmp_h_t.clone()
        new_tmp_c_t = tmp_c_t.clone()
        new_tmp_ctx = tmp_ctx.clone()
        new_tmp_r_t = tmp_r_t.clone()
        new_tmp_img_attn = tmp_img_attn.clone()
        new_tmp_ctx_attn = tmp_ctx_attn.clone()
        new_tmp_ctx_mask = tmp_ctx_mask.clone()

        new_tmp_seen_feat = tmp_seen_feat.clone()
        new_tmp_question = tmp_question.clone()
        new_tmp_pre_ctx_attend = tmp_pre_ctx_attend.clone()

        new_navigable_index = []
        value = log_valuesf.exp()
        new_value = value.clone()

        if step >= 1:
            # we need these as reference when we fork beams around
            beam_seq_prev = beam_seq[:step].clone()
            beam_seq_logprobs_prev = beam_seq_logprobs[:step].clone()

        next_viewpoint_idx, next_viewpoints, next_headings = [], [], []
        beam_idx, cand_idx = 0, 0

        while beam_idx < beam_size and cand_idx < len(candidates):
            v = candidates[cand_idx]

            # given the candidates, let us take the action on env, obtain new ob, record with new traj
            action = v['c']  # tensor([])

            # if the selected action is not 0, means we picked a navigable direction
            # but, the probability can not be -inf because that was the direction we masked out before
            if action >= 1 and v['p'] != -float('inf'):
                next_viewpoint_idx.append(navigable_index[v['q']][action - 1])
                new_pre_feat[beam_idx] = navigable_feat[v['q'], action, :]
                next_viewpoints.append(viewpoints[v['q']][action])
                beam_seq[step, beam_idx] = v['c']  # c'th word is the continuation, i.e. the navigable direction selected

            else:
                next_viewpoint_idx.append('STAY')
                new_ended[beam_idx] = True
                new_pre_feat[beam_idx] = navigable_feat[v['q'], 0, :]
                next_viewpoints.append(viewpoints[v['q']][0])
                beam_seq[step, beam_idx] = torch.tensor(0).long()
            next_headings.append(tmp_obs[v['q']]['navigableLocations'][next_viewpoints[beam_idx]]['heading'])
            new_navigable_index.append(navigable_index[v['q']])
            new_value[beam_idx] = value[v['q']]

            # fork beam index q into index beam_idx
            if step >= 1:
                beam_seq[:step, beam_idx] = beam_seq_prev[:, v['q']]
                beam_seq_logprobs[:step, beam_idx] = beam_seq_logprobs_prev[:, v['q']]

            # append new end terminal at the end of this beam
            beam_seq_logprobs[step, beam_idx] = v['r']  # the raw logprob here
            beam_logprobs_sum[beam_idx] = v['p']  # the new (sum) logprob along this beam

            # move the old info to its new location
            new_tmp_h_t[beam_idx] = tmp_h_t[v['q']]
            new_tmp_c_t[beam_idx] = tmp_c_t[v['q']]
            new_tmp_r_t[beam_idx] = tmp_r_t[v['q']]
            new_tmp_ctx[beam_idx] = tmp_ctx[v['q']]
            new_tmp_img_attn[beam_idx] = tmp_img_attn[v['q']]
            new_tmp_ctx_attn[beam_idx] = tmp_ctx_attn[v['q']]
            new_tmp_ctx_mask[beam_idx] = tmp_ctx_mask[v['q']]
            new_traj[beam_idx] = copy.deepcopy(traj[v['q']])
            new_tmp_seen_feat[beam_idx] = tmp_seen_feat[v['q']]
            new_tmp_question[beam_idx] = tmp_question[v['q']]
            new_tmp_pre_ctx_attend[beam_idx] = tmp_pre_ctx_attend[v['q']]

            beam_idx += 1
            cand_idx += 1

        if len(next_viewpoints) < beam_size:
            # if avoiding the visited states, we might not have enough candidates for number of beams
            # in this case, we replicate from the existed beams
            replicate_idx = 0

            missing_length = beam_size - len(next_viewpoints)
            for _ in range(missing_length):
                next_viewpoints.append(next_viewpoints[replicate_idx])
                new_navigable_index.append(new_navigable_index[replicate_idx])
                next_headings.append(next_headings[replicate_idx])
                next_viewpoint_idx.append(next_viewpoint_idx[replicate_idx])

            beam_seq[:, -missing_length:] = beam_seq[:, replicate_idx].unsqueeze(1).expand(-1, missing_length)
            beam_seq_logprobs[:, -missing_length:] = beam_seq_logprobs[:, replicate_idx].unsqueeze(1).expand(-1, missing_length)
            beam_logprobs_sum[-missing_length:] = beam_logprobs_sum[replicate_idx].expand(missing_length)

            new_ended[-missing_length:] = [new_ended[replicate_idx]] * missing_length
            new_last_recorded[-missing_length:] = [new_last_recorded[replicate_idx]] * missing_length

            new_pre_feat[-missing_length:] = new_pre_feat[replicate_idx].unsqueeze(0).expand(missing_length, -1)
            new_tmp_h_t[-missing_length:] = new_tmp_h_t[replicate_idx].unsqueeze(0).expand(missing_length, -1)
            new_tmp_c_t[-missing_length:] = new_tmp_c_t[replicate_idx].unsqueeze(0).expand(missing_length, -1)
            new_tmp_r_t[-missing_length:] = new_tmp_r_t[replicate_idx].unsqueeze(0).expand(missing_length, -1)
            new_tmp_ctx[-missing_length:] = new_tmp_ctx[replicate_idx].unsqueeze(0).expand(missing_length, -1, -1)
            new_tmp_img_attn[-missing_length:] = new_tmp_img_attn[replicate_idx].unsqueeze(0).expand(missing_length, -1)
            new_tmp_ctx_attn[-missing_length:] = new_tmp_ctx_attn[replicate_idx].unsqueeze(0).expand(missing_length, -1)
            new_tmp_ctx_mask[-missing_length:] = new_tmp_ctx_mask[replicate_idx].unsqueeze(0).expand(missing_length, -1)

            new_value[-missing_length:] = new_value[replicate_idx].unsqueeze(0).expand(missing_length, -1)

        # move within the env
        new_tmp_obs = self.env.teleport_beam(batch_idx, scan_id, next_viewpoints, next_headings)

        for vix, ob in enumerate(new_tmp_obs):
            if not new_ended[vix] or not new_last_recorded[vix]:
                new_traj[vix]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
                dist = super(PanoSeq2SeqAgent, self)._get_distance(ob)
                new_traj[vix]['distance'].append(dist)
                new_traj[vix]['img_attn'].append(new_tmp_img_attn[vix].detach().cpu().numpy().tolist())
                new_traj[vix]['ctx_attn'].append(new_tmp_ctx_attn[vix].detach().cpu().numpy().tolist())
                new_traj[vix]['value'].append(new_value[vix].detach().cpu().item())
                new_traj[vix]['navigable_idx'].append(new_navigable_index[vix])
                new_traj[vix]['viewpoint_idx'].append(next_viewpoint_idx[vix])
                new_traj[vix]['length'] += 1
                new_last_recorded[vix] = True if new_ended[vix] else False

        state = new_tmp_obs, new_traj, new_ended, new_last_recorded, new_pre_feat, new_tmp_h_t, new_tmp_c_t, \
                new_tmp_ctx, new_tmp_img_attn, new_tmp_ctx_attn, new_tmp_ctx_mask, new_tmp_seen_feat, new_tmp_question, new_tmp_pre_ctx_attend, new_tmp_r_t

        return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates

    def sample_beam(self, beam_size=5):
        obs = np.array(self.env.reset())  # load a mini-batch
        batch_size = len(obs)
        sentence = []
        token_num = 0
        if self.opts.use_configuration:
            # batch_configurations, configuration_num_list = super(PanoSeq2SeqAgent, self)._config_batch(obs)
            # ctx, ctx_mask= self.encoder(batch_configurations, configuration_num_list)
            # state_attention_0, r0, h0, c0 = self.encoder.init_state(batch_size, max(configuration_num_list))
            # ctx_attn = state_attention_0
            # h_t = h0
            # c_t = c0
            # r_t = r0
            seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs)
            for ob in obs:
                sentence.append(" quan ".join(ob['configurations']) + " quan")
            tmp_ctx, h_t, c_t, tmp_ctx_mask, split_index = self.encoder(sentence, seq_lengths)
            for each_split in split_index:
                if each_split[-1] > token_num:
                    token_num = each_split[-1]
            batch_configurations, configuration_num_list = super(PanoSeq2SeqAgent, self)._config_batch(obs)
            s0, r0, h0, c0 = self.encoder.init_state(batch_size, max(configuration_num_list))
            seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs)
            new_ctx = torch.zeros(batch_size, 15, token_num, 512, device = tmp_ctx.device)
            new_ctx_mask = torch.zeros(batch_size, 15, token_num, device = tmp_ctx.device)
            new_cls = torch.zeros(batch_size, 15, 512, device = tmp_ctx.device)
            new_cls_mask = torch.zeros(batch_size, 15, device = tmp_ctx.device)
 
            for ob_id, each_index_list in enumerate(split_index):
                start = 0
                for list_id, each_index in enumerate(each_index_list):
                    end = each_index
                    new_ctx[ob_id,list_id,0:end-start,:] = tmp_ctx[ob_id,start:end,:]
                    new_ctx_mask[ob_id, list_id,0:end-start] = 1
                    new_cls[ob_id, list_id, :] = tmp_ctx[ob_id, each_index,:]
                    new_cls_mask[ob_id, list_id] = 1
                    start = end + 1
            weighted_new_ctx, attn = self.encoder.sf(new_cls, new_cls_mask, new_ctx, new_ctx_mask)
            ctx = weighted_new_ctx
            ctx_mask = new_cls_mask    
            ctx_attn = s0
            r_t = r0

        else:
            seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs)
            ctx, h_t, c_t, ctx_mask = self.encoder(seq, seq_lengths)
            tmp_tensor = torch.zeros(5,1)
            tmp_r_t = tmp_tensor

        self.done_beams = [[] for _ in range(batch_size)]
        batch_traj = []

        # we perform beam search within each of the sample inside mini-batch
        for k in range(batch_size):

            # save the visited state to avoid the agent search to a visited state
            self.visited_state = []

            # save all visited states for test server submission
            self.all_visited_traj = []

            # take the sample in mini-batch and duplicate for number of beams
            tmp_obs = np.repeat(obs[k:k+1], beam_size)
            tmp_ctx = ctx[k:k+1].expand(beam_size, ctx.size(1), ctx.size(2))
            tmp_h_t = h_t[k:k + 1].expand(beam_size, h_t.size(1))
            tmp_c_t = c_t[k:k + 1].expand(beam_size, c_t.size(1))
            tmp_ctx_mask = ctx_mask[k:k + 1].expand(beam_size, ctx_mask.size(1))
            
            if self.opts.use_configuration:
                tmp_ctx_attn = ctx_attn[k:k + 1].expand(beam_size, ctx_attn.size(1))
                tmp_r_t = r_t[k:k + 1].expand(beam_size, r_t.size(1))

            tmp_pre_feat = torch.zeros(beam_size, obs[0]['feature'].shape[1]).to(self.device)

            tmp_question = h_t[k:k + 1].expand(beam_size, h_t.size(1))
            tmp_pre_ctx_attend = torch.zeros(beam_size, self.opts.rnn_hidden_size).to(self.device)

            tmp_seen_feat = torch.zeros(beam_size, 1, self.opts.rnn_hidden_size).to(self.device)

            traj, scan_id = [], []
            for ob in tmp_obs:
                traj.append({
                    'instr_id': ob['instr_id'],
                    'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
                    'length': 1,
                    'feature': [ob['feature']],
                    'img_attn': [],
                    'ctx_attn': [],
                    'rollback_forward_attn': [],
                    'value': [],
                    'viewpoint_idx': [],
                    'navigable_idx': [],
                    'gt_viewpoint_idx': ob['gt_viewpoint_idx'],
                    'distance': [super(PanoSeq2SeqAgent, self)._get_distance(ob)]
                })
                scan_id.append(ob['scan'])
                if (ob['viewpoint'], ob['heading']) not in self.visited_state:
                    self.visited_state.append((ob['viewpoint'], ob['heading']))

            ended = np.array([False] * beam_size)
            last_recorded = np.array([False] * beam_size)

            if self.opts.use_configuration:
                viewpoints, navigable_index, navigable_feat, tmp_h_t, tmp_c_t, logit, value, tmp_img_attn, tmp_ctx_attn, \
                tmp_pre_ctx_attend = self.model_seen_step(tmp_obs, ended, tmp_pre_feat, tmp_h_t, tmp_c_t, tmp_ctx, tmp_ctx_mask,
                                                    tmp_question, tmp_pre_ctx_attend, tmp_ctx_attn, tmp_r_t)
            else:
                viewpoints, navigable_index, navigable_feat, tmp_h_t, tmp_c_t, logit, value, tmp_img_attn, tmp_ctx_attn, \
                tmp_pre_ctx_attend = self.model_seen_step(tmp_obs, ended, tmp_pre_feat, tmp_h_t, tmp_c_t, tmp_ctx, tmp_ctx_mask,
                                                    tmp_question, tmp_pre_ctx_attend, tmp_tensor, tmp_tensor)

            logprobs = F.log_softmax(logit, dim=1)
            logprobs_value = ((value + 1) / 2).log()

            self.done_beams[k] = self.beam_search(k, logprobs, tmp_obs, traj, ended, last_recorded,
                                                  viewpoints, navigable_index, navigable_feat, tmp_pre_feat,
                                                  tmp_ctx, tmp_img_attn, tmp_ctx_attn, tmp_h_t,
                                                  tmp_c_t, tmp_ctx_mask, beam_size,
                                                  tmp_seen_feat, tmp_question, tmp_pre_ctx_attend, tmp_r_t, logprobs_value)

            batch_traj.append(self.done_beams[k][0]['traj'])

        return batch_traj

    def beam_search(self, batch_idx, logprobs, tmp_obs, traj, ended, last_recorded, viewpoints, navigable_index,
                    navigable_feat, tmp_pre_feat, tmp_ctx, tmp_img_attn, tmp_ctx_attn, tmp_h_t, tmp_c_t, tmp_ctx_mask,
                    beam_size, tmp_seen_feat, tmp_question, tmp_pre_ctx_attend, tmp_r_t, logprobs_value=None):

        def beam_step_state_factored(visited_state, batch_idx, scan_id, logprobsf, beam_size, step, beam_seq,
                                     beam_seq_logprobs, beam_logprobs_sum, state, logprobsf_value=None):
            if logprobsf_value is not None:
                ys, ix = torch.sort(logprobsf_value.repeat(1, logprobsf.size(1)) + logprobsf, 1, True)
            else:
                ys, ix = torch.sort(logprobsf, 1, True)

            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size

            # if this is the first step, we will just use the 1st row
            # instead of use the number of beams
            if step == 0:
                rows = 1

            for c in range(cols):  # for each column (word, essentially)
                for q in range(rows):  # for each beam expansion
                    # compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q, c]
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    candidates.append({'c': ix[q, c], 'q': q, 'p': candidate_logprob, 'r': local_logprob})
            candidates = sorted(candidates, key=lambda x: -x['p'])

            # let us duplicate the state and then we will more stuffs around
            # tmp_obs, traj, ended, last_recorded, viewpoints, navigable_index, navigable_feat, tmp_pre_feat, tmp_h_t, tmp_c_t, tmp_ctx, tmp_ctx_mask = state
            tmp_obs, traj, ended, last_recorded, viewpoints, navigable_index, navigable_feat, tmp_pre_feat, tmp_h_t, \
            tmp_c_t, tmp_ctx, tmp_img_attn, tmp_ctx_attn, tmp_ctx_mask, tmp_seen_feat, tmp_question, tmp_pre_ctx_attend, tmp_r_t = state

            new_traj = copy.deepcopy(traj)
            new_ended = ended.copy()
            new_last_recorded = last_recorded.copy()
            new_pre_feat = tmp_pre_feat.clone()
            new_tmp_h_t = tmp_h_t.clone()
            new_tmp_c_t = tmp_c_t.clone()
            new_tmp_ctx = tmp_ctx.clone()
            new_tmp_r_t = tmp_r_t.clone()
            new_tmp_img_attn = tmp_img_attn.clone()
            new_tmp_ctx_attn = tmp_ctx_attn.clone()
            new_tmp_ctx_mask = tmp_ctx_mask.clone()

            new_tmp_seen_feat = tmp_seen_feat.clone()
            new_tmp_question = tmp_question.clone()
            new_tmp_pre_ctx_attend = tmp_pre_ctx_attend.clone()

            new_navigable_index = []
            value = logprobsf_value.exp()
            new_value = value.clone()

            if step >= 1:
                # we need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:step].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:step].clone()

            next_viewpoint_idx, next_viewpoints, next_headings = [], [], []

            beam_idx, cand_idx = 0, 0

            num_candidate_routes = 1 if self.opts.progress_inference else beam_size
            while beam_idx < num_candidate_routes and cand_idx < len(candidates):
                v = candidates[cand_idx]

                # given the candidates, let us take the action on env, obtain new ob, record with new traj
                action = v['c']  # tensor([])

                # if the selected action is not 0, means we picked a navigable direction
                # but, the probability can not be -inf because that was the direction we masked out before
                if action >= 1 and v['p'] != -float('inf'):

                    next_viewpoint = viewpoints[v['q']][action]
                    next_heading = tmp_obs[v['q']]['navigableLocations'][next_viewpoint]['heading']

                    # let's check if the action we are going to take will lead to a state visited before
                    if (next_viewpoint, next_heading) not in visited_state:
                    # if next_viewpoint not in visited_state:
                        visited_state.append((next_viewpoint, next_heading))

                        next_viewpoint_idx.append(navigable_index[v['q']][action - 1])
                        new_pre_feat[beam_idx] = navigable_feat[v['q'], action, :]
                        next_viewpoints.append(viewpoints[v['q']][action])
                        beam_seq[step, beam_idx] = v['c']  # c'th word is the continuation, i.e. the navigable direction selected
                    else:
                        cand_idx += 1
                        continue
                else:
                    next_viewpoint_idx.append('STAY')
                    new_ended[beam_idx] = True
                    new_pre_feat[beam_idx] = navigable_feat[v['q'], 0, :]
                    next_viewpoints.append(viewpoints[v['q']][0])
                    beam_seq[step, beam_idx] = torch.tensor(0).long()
                next_headings.append(tmp_obs[v['q']]['navigableLocations'][next_viewpoints[beam_idx]]['heading'])
                new_navigable_index.append(navigable_index[v['q']])
                new_value[beam_idx] = value[v['q']]

                # fork beam index q into index beam_idx
                if step >= 1:
                    beam_seq[:step, beam_idx] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:step, beam_idx] = beam_seq_logprobs_prev[:, v['q']]

                # append new end terminal at the end of this beam
                beam_seq_logprobs[step, beam_idx] = v['r']  # the raw logprob here
                beam_logprobs_sum[beam_idx] = v['p']  # the new (sum) logprob along this beam

                # move the old info to its new location
                new_tmp_h_t[beam_idx] = tmp_h_t[v['q']]
                new_tmp_c_t[beam_idx] = tmp_c_t[v['q']]
                new_tmp_r_t[beam_idx] = tmp_r_t[v['q']]
                new_tmp_ctx[beam_idx] = tmp_ctx[v['q']]
                new_tmp_img_attn[beam_idx] = tmp_img_attn[v['q']]
                new_tmp_ctx_attn[beam_idx] = tmp_ctx_attn[v['q']]
                new_tmp_ctx_mask[beam_idx] = tmp_ctx_mask[v['q']]

                new_ended[beam_idx] = ended[v['q']]
                new_last_recorded[beam_idx] = last_recorded[v['q']]
                new_traj[beam_idx] = copy.deepcopy(traj[v['q']])

                new_tmp_seen_feat[beam_idx] = tmp_seen_feat[v['q']]
                new_tmp_question[beam_idx] = tmp_question[v['q']]
                new_tmp_pre_ctx_attend[beam_idx] = tmp_pre_ctx_attend[v['q']]

                beam_idx += 1
                cand_idx += 1

            if len(next_viewpoints) < beam_size:
                # if avoiding the visited states, we might not have enough candidates for number of beams
                # in this case, we replicate from the existed beams
                replicate_idx = 0

                missing_length = beam_size - len(next_viewpoints)
                for _ in range(missing_length):
                    next_viewpoints.append(next_viewpoints[replicate_idx])
                    new_navigable_index.append(new_navigable_index[replicate_idx])
                    next_headings.append(next_headings[replicate_idx])
                    next_viewpoint_idx.append(next_viewpoint_idx[replicate_idx])

                beam_seq[:, -missing_length:] = beam_seq[:, replicate_idx].unsqueeze(1).expand(-1, missing_length)
                beam_seq_logprobs[:, -missing_length:] = beam_seq_logprobs[:, replicate_idx].unsqueeze(1).expand(-1, missing_length)
                beam_logprobs_sum[-missing_length:] = torch.zeros(missing_length).fill_(-float('inf'))

                new_ended[-missing_length:] = [new_ended[replicate_idx]] * missing_length
                new_last_recorded[-missing_length:] = [new_last_recorded[replicate_idx]] * missing_length

                new_pre_feat[-missing_length:] = new_pre_feat[replicate_idx].unsqueeze(0).expand(missing_length, -1)
                new_tmp_h_t[-missing_length:] = new_tmp_h_t[replicate_idx].unsqueeze(0).expand(missing_length, -1)
                new_tmp_c_t[-missing_length:] = new_tmp_c_t[replicate_idx].unsqueeze(0).expand(missing_length, -1)
                new_tmp_r_t[-missing_length:] = new_tmp_r_t[replicate_idx].unsqueeze(0).expand(missing_length, -1)
                new_tmp_ctx[-missing_length:] = new_tmp_ctx[replicate_idx].unsqueeze(0).expand(missing_length, -1, -1)
                new_tmp_img_attn[-missing_length:] = new_tmp_img_attn[replicate_idx].unsqueeze(0).expand(missing_length, -1)
                new_tmp_ctx_attn[-missing_length:] = new_tmp_ctx_attn[replicate_idx].unsqueeze(0).expand(missing_length, -1)
                new_tmp_ctx_mask[-missing_length:] = new_tmp_ctx_mask[replicate_idx].unsqueeze(0).expand(missing_length, -1)

                new_value[-missing_length:] = new_value[replicate_idx].unsqueeze(0).expand(missing_length, -1)

            # move within the env
            new_tmp_obs = self.env.teleport_beam(batch_idx, scan_id, next_viewpoints, next_headings)

            for vix, ob in enumerate(new_tmp_obs):
                if not new_ended[vix]:
                    new_traj[vix]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
                    dist = super(PanoSeq2SeqAgent, self)._get_distance(ob)
                    new_traj[vix]['distance'].append(dist)
                    new_traj[vix]['img_attn'].append(new_tmp_img_attn[vix].detach().cpu().numpy().tolist())
                    new_traj[vix]['ctx_attn'].append(new_tmp_ctx_attn[vix].detach().cpu().numpy().tolist())
                    new_traj[vix]['value'].append(new_value[vix].detach().cpu().item())
                    new_traj[vix]['navigable_idx'].append(new_navigable_index[vix])
                    new_traj[vix]['viewpoint_idx'].append(next_viewpoint_idx[vix])
                    new_traj[vix]['length'] += 1
                    new_last_recorded[vix] = True if new_ended[vix] else False

            state = new_tmp_obs, new_traj, new_ended, new_last_recorded, new_pre_feat, new_tmp_h_t, new_tmp_c_t, \
                    new_tmp_ctx, new_tmp_img_attn, new_tmp_ctx_attn, new_tmp_ctx_mask, new_tmp_seen_feat, new_tmp_question, new_tmp_pre_ctx_attend, new_tmp_r_t

            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates

        beam_seq = torch.zeros(self.opts.max_episode_len, beam_size).to(self.device)
        beam_seq_logprobs = torch.zeros(self.opts.max_episode_len, beam_size).to(self.device)
        beam_logprobs_sum = torch.zeros(beam_size).to(self.device) # running sum of logprobs for each beam
        done_beams = []

        scan_id = [_['scan'] for _ in tmp_obs]

        for step in range(self.opts.max_episode_len):
            logprobsf = logprobs.data
            if logprobs_value is not None:
                logprobsf_value = logprobs_value.data

            state = tmp_obs, traj, ended, last_recorded, viewpoints, navigable_index, navigable_feat, tmp_pre_feat, \
                    tmp_h_t, tmp_c_t, tmp_ctx, tmp_img_attn, tmp_ctx_attn, tmp_ctx_mask, tmp_seen_feat, tmp_question, tmp_pre_ctx_attend, tmp_r_t

            beam_seq, \
            beam_seq_logprobs, \
            beam_logprobs_sum, \
            state, \
            candidates_divm = beam_step_state_factored(self.visited_state, batch_idx, scan_id, logprobsf, beam_size,
                                                       step, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state,
                                                       logprobsf_value)

            # get the new sorted state
            tmp_obs, traj, ended, last_recorded, tmp_pre_feat, tmp_h_t, tmp_c_t, tmp_ctx, tmp_img_attn, tmp_ctx_attn, \
            tmp_ctx_mask, tmp_seen_feat, tmp_question, tmp_pre_ctx_attend, tmp_r_t = state

            for vix in range(beam_size):

                current_path = traj[vix]['path']

                # save all the visited states for submitting to the test server
                if current_path not in self.all_visited_traj and beam_logprobs_sum[vix] != -float('inf') :
                    if self.all_visited_traj != []:
                        insert_idx_set = []

                        # for all past trajs, we check one-by-one for each viewpoint, to see if there is
                        # overlap with the current path
                        for vs_idx, visited_states in enumerate(self.all_visited_traj):

                            insert_idx = -1
                            for vp_idx, (viewpoint, _, _) in enumerate(visited_states):
                                if viewpoint == current_path[vp_idx][0]:
                                    insert_idx = vp_idx

                            insert_idx_set.append((vs_idx, insert_idx))

                        # find the farthest overlapping viewpoint and insert the current path next to this traj
                        vs_insert_idx, max_insert_idx, = max(insert_idx_set, key=lambda item: item[1])
                        self.all_visited_traj.insert(vs_insert_idx, current_path)

                        if current_path not in self.all_visited_traj:
                            self.all_visited_traj.append(current_path)
                    else:
                        self.all_visited_traj.append(current_path)

                # if time's up... or if end token is reached then copy beams
                if beam_logprobs_sum[vix] != -float('inf'):
                    if beam_seq[step, vix] == 0 or step == self.opts.max_episode_len - 1:
                        final_beam = {
                            'seq': beam_seq[:, vix].clone(),
                            'logps': beam_seq_logprobs[:, vix].clone(),
                            'p': beam_logprobs_sum[vix].clone(),
                            'traj': traj[vix]
                        }
                        done_beams.append(final_beam)
                        # don't continue beams from finished sequences
                        beam_logprobs_sum[vix] = -float('inf')

            viewpoints, navigable_index, navigable_feat, tmp_h_t, tmp_c_t, logit, value, tmp_img_attn, tmp_ctx_attn, \
            tmp_pre_ctx_attend = self.model_seen_step(tmp_obs, ended, tmp_pre_feat, tmp_h_t, tmp_c_t,
                                         tmp_ctx, tmp_ctx_mask, tmp_question, tmp_pre_ctx_attend, tmp_ctx_attn, tmp_r_t)

            logprobs = F.log_softmax(logit, dim=1)
            logprobs_value = ((value + 1) / 2).log()

        # sort beams based on their accumulated prob
        done_beams = sorted(done_beams, key=lambda x: -x['p'])

        # append the selected final traj to the last
        self.all_visited_traj.append(done_beams[0]['traj']['path'])

        return done_beams[:beam_size]




    