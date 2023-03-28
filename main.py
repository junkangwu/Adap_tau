# from ast import arg
import argparse
import os
import random

import torch
import numpy as np
from utils.parser import parse_args

import time, json, sys, os
import logging, logging.config
from tqdm import tqdm
from copy import deepcopy
import logging
# from prettytable import PrettyTable
from torch_scatter import scatter
from utils.data_loader import load_data
from utils.evaluate import test_sp
import torch.nn.functional as F
import os.path as osp
n_users = 0
n_items = 0
def get_logger(name, log_dir, config_dir):
    """
    Creates a logger object

    Parameters
    ----------
    name:           Name of the logger file
    log_dir:        Directory where logger file needs to be stored
    config_dir:     Directory from where log_config.json needs to be read
    
    Returns
    -------
    A logger object which writes to both file and stdout
        
    """
    config_dict = json.load(open( config_dir + 'log_config.json'))
    config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger

class Sample(object):
    def __init__(self, user_dict, n_users, n_items, sampling_method="uniform", train_cf = None):
        self.n_users = n_users
        self.n_items = n_items
        self.random_list = []
        self.random_pr = 0
        self.random_list_length = 0
        self.sampling_method = sampling_method

        if self.sampling_method == "neg" :
            self.set_distribution(train_cf)
            self.used_ids = np.array([set() for _ in range(n_users)])
            for user in user_dict['train_user_set']:
                self.used_ids[user] = set(user_dict['train_user_set'][user])
        elif self.sampling_method == "uniform_gpu":
            self.p_sample_1 = torch.ones((args.batch_size, self.n_items), device=device)
            self.p_sample_2 = torch.ones((len(train_cf) % args.batch_size, self.n_items), device=device)

    def set_distribution(self, train_cf=None):
        """Set the distribution of sampler.

        Args:
            distribution (str): Distribution of the negative items.
        """
        if self.sampling_method == "neg":
            self.random_list = np.arange(self.n_items)
            np.random.shuffle(self.random_list)
            self.random_pr = 0
            self.random_list_length = len(self.random_list)

        elif self.sampling_method == "pop":
            self.random_list = train_cf[:, 1]
            np.random.shuffle(self.random_list)
            
    def random_num(self, num):
        value_id = []
        self.random_pr %= self.random_list_length
        cnt = 0
        while True:
            if self.random_pr + num <= self.random_list_length:
                value_id.append(self.random_list[self.random_pr: self.random_pr + num])
                self.random_pr += num
                break
            else:
                value_id.append(self.random_list[self.random_pr:])
                num -= self.random_list_length - self.random_pr
                self.random_pr = 0
                cnt += 1
        return np.concatenate(value_id)

    def get_sample_by_key_ids(self, key_ids, num):
        key_ids = np.array(key_ids.cpu().numpy())
        key_num = len(key_ids)
        total_num = key_num * num
        # start
        value_ids = np.zeros(total_num, dtype=np.int64)
        check_list = np.arange(total_num)
        key_ids = np.tile(key_ids, num)
        # cnt = 0
        while len(check_list) > 0:
            value_ids[check_list] = self.random_num(len(check_list))
            check_list = np.array([
                    i for i, used, v in zip(check_list, self.used_ids[key_ids[check_list]], value_ids[check_list])
                    if v in used
                ])
        value_ids = torch.LongTensor(value_ids).to(device).view(-1, key_num) # [M, B]
        value_ids = value_ids.t().contiguous() # [B, M]

        return value_ids[:, :K]
        
    def get_feed_dict(self, train_entity_pairs, train_pos_set, start, end, n_negs=1):
        feed_dict = {}
        entity_pairs = train_entity_pairs[start: end]
        feed_dict['users'] = entity_pairs[:, 0]
        feed_dict['pos_items'] = entity_pairs[:, 1]
        if self.sampling_method == "uniform":
            neg_items = np.random.choice(self.n_items, size=(entity_pairs.shape[0], n_negs),
                                        replace=True)
            feed_dict['neg_items'] = torch.LongTensor(neg_items).to(device)
        elif self.sampling_method == "uniform_gpu":
            neg_items = torch.multinomial(self.p_sample_1, num_samples=n_negs, replacement=True)
            feed_dict['neg_items'] = neg_items
        elif self.sampling_method == "neg":
            feed_dict['neg_items'] = self.get_sample_by_key_ids(entity_pairs[:, 0], n_negs*K)
        elif self.sampling_method == "no_sample":
            return feed_dict
    
        return feed_dict

    def get_feed_dict_reset(self, train_entity_pairs, train_pos_set, start, n_negs=1):
        feed_dict = {}
        entity_pairs = train_entity_pairs[start:]
        feed_dict['users'] = entity_pairs[:, 0]
        feed_dict['pos_items'] = entity_pairs[:, 1]
        if self.sampling_method == "uniform":
            neg_items = np.random.choice(self.n_items, size=(entity_pairs.shape[0], n_negs),
                                        replace=True)
            feed_dict['neg_items'] = torch.LongTensor(neg_items).to(device)
        elif self.sampling_method == "uniform_gpu":
            neg_items = torch.multinomial(self.p_sample_2, num_samples=n_negs, replacement=True)
            feed_dict['neg_items'] = neg_items
        elif self.sampling_method == "neg":
            feed_dict['neg_items'] = self.get_sample_by_key_ids(entity_pairs[:, 0], n_negs*K)
        elif self.sampling_method == "no_sample":
            return feed_dict
       
        return feed_dict
  


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device, K
    args = parse_args()
    # print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    if not args.restore: 
        args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')
    
    logger = get_logger(args.name, args.log_dir, args.config_dir)
    logger.info(vars(args))
    """build dataset"""
    train_cf, user_dict, sp_matrix, n_params, norm_mat, valid_pre, test_pre, item_group_idx = load_data(args, logger=logger)

    train_cf_size = len(train_cf)

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_negs = args.n_negs
    # K = args.K
    args.Ks = eval(args.Ks)

    sample = Sample(user_dict, n_users, n_items, sampling_method=args.sampling_method, train_cf=train_cf)
    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    """define model"""
    from modules.MF_tau import MF
    from modules.LGN_tau import lgn_frame
    if args.gnn == 'mf':
        model = MF(n_params, args, norm_mat, logger).to(device)
    elif args.gnn == "lgn":
        model = lgn_frame(n_params, args, norm_mat, logger).to(device)
    else:
        raise NotImplementedError
    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    kill_cnt = 0
    best_ndcg = -np.inf
    eval_earlystop = args.eval_earlystop.split('@')
    eval_to_int = {'ndcg':0, 'recall':1, 'precision':2}
    eval_str = [eval_to_int[eval_earlystop[0]], eval(eval_earlystop[1])]
    logger.info('Evaluation Protocols is {} @ {}'.format(eval_str[0], eval_str[1]))
    """ makdir weights dir"""
    args.out_dir = os.path.join(args.out_dir, args.dataset)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not args.restore:
        logger.info("start training ...")
        loss_per_user = None
        loss_per_ins = None
        # prepare for tau_0
        pos = train_cf.to(device)
        nu = scatter(torch.ones(len(train_cf), device=device), pos[:, 0], dim=0, reduce='sum')
        nu_thresh = torch.quantile(nu, 0.2)
        judgeid_torch = (nu > nu_thresh)
        [useid_torch, ] = torch.where(judgeid_torch > 0)
        [yid_torch ,] = torch.where(judgeid_torch[pos[:,0]]>0)

        for epoch in range(args.epoch):
            train_cf_ = train_cf
            index = np.arange(len(train_cf_))
            np.random.shuffle(index)
            train_cf_ = train_cf_[index].to(device)

            """training"""
            model.train()
            loss, s = 0, 0
            losses_train = []
            tau_maxs = []
            tau_mins = []
            losses_emb = []
            hits = 0
            train_s_t = time.time()
    
            if epoch >= args.cnt_lr:
                user_emb_cos, item_emb_cos = model.gcn_emb()

                user_emb_cos = F.normalize(user_emb_cos, dim=-1)
                item_emb_cos = F.normalize(item_emb_cos, dim=-1)

                pos_scores = (user_emb_cos[pos[:, 0]] * item_emb_cos[pos[:, 1]]).sum(dim=-1)
                pos_u_torch = pos_scores[yid_torch].mean()
                # pos_var_torch = pos_scores[yid_torch].var()
                ev_mean_torch = item_emb_cos.mean(dim=0, keepdim=True)
                allu_torch = (user_emb_cos[useid_torch] @ ev_mean_torch.t()).view(-1)
                au_torch = allu_torch.mean()
                can_torch = np.log(len(useid_torch) * n_items)
                a_torch = 1e-10
                c_torch = 2 * (np.log(0.5)+can_torch-np.log(len(yid_torch)))
                b_torch = - (pos_u_torch - au_torch)
                # w_torch = c_torch / (-2 * b_torch)
                w_0 = c_torch / (-2 * b_torch)
                logger.info("current w_0 is {}".format(w_0.item()))
            else:
                can = np.log(len(useid_torch) * n_items);
                a = 1e-10;
                c = 2 * (np.log(0.5) + can - np.log(len(yid_torch)))
                print(c / 2)
                b = - 0.7
                w_0 = ( - b - np.sqrt(np.clip(b ** 2 - a*c , 0, 100000))) / a
                logger.info("current w_0 is {}".format(w_0))
                    # loss_per_user = scatter(losses_train, train_cf_[:, 0], dim=0, reduce='mean')

            while s + args.batch_size <= len(train_cf):
                # print('Step: {}'.format(s))
                batch = sample.get_feed_dict(train_cf_,
                                      user_dict['train_user_set'],
                                      s, s + args.batch_size,
                                      n_negs)

                batch_loss, train_loss, emb_loss, tau = model(batch, loss_per_user=loss_per_user, w_0=w_0, s=s)
                tau_maxs.append(tau.max().item())
                tau_mins.append(tau.min().item())
                losses_emb.append(emb_loss.item())
                losses_train.append(train_loss)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss.item()
                s += args.batch_size
            
            # reset pairs training
            if len(train_cf) - s < args.batch_size:
                batch = sample.get_feed_dict_reset(train_cf_,
                                      user_dict['train_user_set'],
                                      s, n_negs)
                batch_loss, train_loss, emb_loss, tau = model(batch, loss_per_user=loss_per_user, w_0=w_0, s=s)
                tau_maxs.append(tau.max().item())
                tau_mins.append(tau.min().item())
                losses_emb.append(emb_loss.item())
                losses_train.append(train_loss)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss.item()
                s += args.batch_size
            train_e_t = time.time()
            
            losses_train = torch.cat(losses_train, dim=0)
            loss_per_user = scatter(losses_train, train_cf_[:, 0], dim=0, reduce='mean')
            # valid
            model.eval()
            with torch.no_grad():
                valid_st = time.time()
                valid_ret = test_sp(model, user_dict, sp_matrix, n_params, valid_pre, test_pre, mode='valid')
                test_ret = test_sp(model, user_dict, sp_matrix, n_params, valid_pre, test_pre, mode='test')
                valid_ed = time.time()
            print_result = 'E:{}|TAU:{:.4} {:.4}, train_time: {:.4}, VALID_time: {:.4}, loss: {:.4}, emb_loss:{:.4}, best_valid({}): {:.4}\n'.format(epoch, 
                        np.mean(tau_mins), np.mean(tau_maxs), train_e_t - train_s_t, valid_ed - valid_st, loss, np.mean(losses_emb), args.eval_earlystop, best_ndcg)
            for k in args.Ks:
                print_result += 'valid \t N@{}: {:.4}, R@{}: {:.4}, P@{}: {:.4}\n'.format(
                    k, valid_ret[0][k-1], k, valid_ret[1][k-1], k, valid_ret[2][k-1])
            logger.info(print_result)
            
            if valid_ret[eval_str[0]][eval_str[1] - 1] > best_ndcg:
                best_ndcg = valid_ret[eval_str[0]][eval_str[1] - 1]
                kill_cnt = 0
                save_path = os.path.join(args.out_dir,  args.name + '.ckpt')
                torch.save(model.state_dict(), save_path)
            else:
                kill_cnt += 1
                if kill_cnt > 50:
                    break
    # test
    logger.info('start to test!!\n')
    load_path = os.path.join(args.out_dir,  args.name + '.ckpt')
    model.load_state_dict(torch.load(load_path), False)
    model.eval()
    with torch.no_grad():
        test_ret = test_sp(model, user_dict, sp_matrix, n_params, valid_pre, test_pre, item_group_idx, mode='test')

    # logger.info('Test result: NDCG@20: {:.4} Recall@20: {:.4}'.format(test_ret[0], test_ret[1]))
    print_result = '\n'
    for k in args.Ks:
        print_result += 'TEST \t N@{}: {:.4}, R@{}: {:.4}, P@{}: {:.4}\n'.format(
            k, test_ret[0][k-1], k, test_ret[1][k-1], k, test_ret[2][k-1])
    logger.info(print_result)



