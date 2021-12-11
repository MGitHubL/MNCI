import sys
import math
import torch
import ctypes
import datetime
import numpy as np

from collections import Counter
from torch.autograd import Variable
from dataset import MNCIDataSet
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

FType = torch.FloatTensor
LType = torch.LongTensor

DID = 0

ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')


class MNCI:
    def __init__(self, directed=False):
        self.network = 'bitalpha'
        self.file_path = '../../data/%s/%s.txt' % (self.network, self.network)
        self.emb_path = '../emb/%s/%s_MNCI_%d.emb'

        self.emb_size = 128
        self.neg_size = 10
        self.hist_len = 128
        self.lr = 0.001
        self.batch = 128
        self.save_step = 5
        self.epochs = 5  # epoch = 5/10
        self.community_number = 10

        self.data = MNCIDataSet(self.file_path, self.neg_size, self.hist_len, self.emb_size, directed)

        self.node_dim = self.data.get_node_dim()
        self.first_time = self.data.get_first_time()

        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                self.node_emb = Variable(MNCI.position_encoding_(self, self.node_dim, self.emb_size).cuda(),
                                         requires_grad=False)
                self.com_emb = Variable(MNCI.position_encoding_(self, self.community_number,
                                                                self.emb_size).type(FType).cuda(), requires_grad=False)

                self.delta_co = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)
                self.delta_co = MNCI.truncated_normal_(self.delta_co, tensor=(self.delta_co))
                self.delta_ne = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)
                self.delta_ne = MNCI.truncated_normal_(self.delta_ne, tensor=(self.delta_ne))

                self.w_node = Variable((torch.zeros(4, self.emb_size, self.emb_size) + 1.).type(FType).cuda(),
                                       requires_grad=True)
                self.w_node = MNCI.truncated_normal_(self.w_node, tensor=(self.w_node))
                self.w_neighbor = Variable((torch.zeros(4, self.emb_size, self.emb_size) + 1.).type(FType).cuda(),
                                           requires_grad=True)
                self.w_neighbor = MNCI.truncated_normal_(self.w_neighbor, tensor=(self.w_neighbor))
                self.w_community = Variable((torch.zeros(4, self.emb_size, self.emb_size) + 1.).type(FType).cuda(),
                                            requires_grad=True)
                self.w_community = MNCI.truncated_normal_(self.w_community, tensor=(self.w_community))
                self.b = Variable((torch.zeros(4, self.node_dim, self.emb_size) + 1.).type(FType).cuda(),
                                  requires_grad=True)
                self.b = MNCI.truncated_normal_(self.b, tensor=(self.b))

                self.time_omega = Variable((torch.zeros(self.emb_size // 2) + 1.).type(FType).cuda(), requires_grad=True)
                self.time_omega = MNCI.truncated_normal_(self.time_omega, tensor=(self.time_omega))

        else:
            self.node_emb = Variable(MNCI.position_encoding_(self, self.node_dim, self.emb_size), requires_grad=False)
            self.com_emb = Variable(MNCI.position_encoding_(self, self.community_number,
                                                            self.emb_size).type(FType), requires_grad=False)

            self.delta_co = Variable((torch.zeros(self.node_dim) + 1.).type(FType), requires_grad=True)
            self.delta_co = MNCI.truncated_normal_(self.delta_co, tensor=(self.delta_co))
            self.delta_ne = Variable((torch.zeros(self.node_dim) + 1.).type(FType), requires_grad=True)
            self.delta_ne = MNCI.truncated_normal_(self.delta_ne, tensor=(self.delta_ne))

            self.w_node = Variable((torch.zeros(4, self.emb_size, self.emb_size) + 1.).type(FType), requires_grad=True)
            self.w_node = MNCI.truncated_normal_(self.w_node, tensor=(self.w_node))
            self.w_neighbor = Variable((torch.zeros(4, self.emb_size, self.emb_size) + 1.).type(FType),
                                       requires_grad=True)
            self.w_neighbor = MNCI.truncated_normal_(self.w_neighbor, tensor=(self.w_neighbor))
            self.w_community = Variable((torch.zeros(4, self.emb_size, self.emb_size) + 1.).type(FType),
                                        requires_grad=True)
            self.w_community = MNCI.truncated_normal_(self.w_community, tensor=(self.w_community))
            self.b = Variable((torch.zeros(4, self.node_dim, self.emb_size) + 1.).type(FType),
                              requires_grad=True)
            self.b = MNCI.truncated_normal_(self.b, tensor=(self.b))

            self.time_omega = Variable((torch.zeros(self.emb_size // 2) + 1.).type(FType), requires_grad=True)
            self.time_omega = MNCI.truncated_normal_(self.time_omega, tensor=(self.time_omega))

        self.opt = torch.optim.Adam(lr=self.lr, params=[self.node_emb, self.delta_co, self.delta_ne, self.w_node,
                                                        self.w_neighbor, self.w_community, self.b, self.com_emb,self.time_omega])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=0.98)

        self.loss = torch.FloatTensor()

    def position_encoding_(self, max_len, emb_size):
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def truncated_normal_(self, tensor, mean=0, std=0.09):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def the_GRU(self, s_nodes, node_emb, neighborhood_inf, community_inf):
        w_node = self.w_node
        w_neighbor = self.w_neighbor
        w_community = self.w_community
        b = self.b.index_select(1, Variable(s_nodes.view(-1)))

        U_G = torch.sigmoid(torch.mm(node_emb, w_node[0]) + torch.mm(neighborhood_inf, w_neighbor[0]) +
                            torch.mm(community_inf, w_community[0]) + b[0])
        NR_G = torch.sigmoid(torch.mm(node_emb, w_node[1]) + torch.mm(neighborhood_inf, w_neighbor[1]) +
                             torch.mm(community_inf, w_community[1]) + b[1])
        CR_G = torch.sigmoid(torch.mm(node_emb, w_node[2]) + torch.mm(neighborhood_inf, w_neighbor[2]) +
                             torch.mm(community_inf, w_community[2]) + b[2])
        tem_node_emb = torch.tanh(torch.mm(node_emb, w_node[3]) +
                                  torch.mul(NR_G, torch.mm(neighborhood_inf, w_neighbor[3])) +
                                  torch.mul(CR_G, torch.mm(community_inf, w_community[3]) + b[3]))
        new_node_emb = torch.mul((1 - U_G), node_emb) + torch.mul(U_G, tem_node_emb)

        return new_node_emb

    def forward(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        batch = s_nodes.size()[0]
        s_node_emb = self.node_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)
        t_node_emb = self.node_emb.index_select(0, Variable(t_nodes.view(-1))).view(batch, -1)
        h_node_emb = self.node_emb.index_select(0, Variable(h_nodes.view(-1))).view(batch, self.hist_len, -1)
        n_node_emb = self.node_emb.index_select(0, Variable(n_nodes.view(-1))).view(batch, self.neg_size, -1)
        community_emb = self.com_emb

        h_time_mask = Variable(h_time_mask)
        h_node_emb = torch.mul(h_node_emb, h_time_mask.unsqueeze(-1))
        # 'h_time_mask' mains that if there is a invalid neighbor in the sequence
        # we need to ensure that it does not play a role in the calculation

        delta_co = self.delta_co.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)
        delta_ne = self.delta_ne.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)
        time_omega = self.time_omega.unsqueeze(0)

        # neighborhood influence
        affinity_per_neighbor = torch.sigmoid(((s_node_emb.unsqueeze(1) - h_node_emb) ** 2).sum(
            dim=-1, keepdim=False).neg()) * h_time_mask
        affinity_sum = affinity_per_neighbor.sum(dim=-1, keepdim=True) + 1e-6
        affinity_weight = affinity_per_neighbor / affinity_sum

        d_time = torch.abs(t_times.unsqueeze(1) - h_times)
        time_emb = Variable(torch.zeros(batch, self.hist_len, self.emb_size).type(FType).cuda())
        middle_value = torch.mul(time_omega.unsqueeze(0), d_time.unsqueeze(-1))
        time_emb[:, :, 0::2] = torch.sin(middle_value)
        time_emb[:, :, 1::2] = torch.cos(middle_value)

        neighborhood_param = torch.mul((affinity_weight * h_time_mask).unsqueeze(-1), time_emb)
        neighborhood_inf = torch.mul(delta_ne, torch.mul(neighborhood_param,
                                                         h_node_emb).sum(dim=1, keepdim=False))

        # community influence
        weight_per_community = torch.sigmoid(((community_emb.unsqueeze(1) - s_node_emb.unsqueeze(0))
                                              ** 2).sum(dim=-1, keepdim=False).neg())
        weight_sum = weight_per_community.sum(dim=0, keepdim=True)
        community_weight = weight_per_community / (weight_sum + 1e-6)
        community_inf = torch.mul(
            delta_co, torch.mul(community_weight.unsqueeze(-1),
                                community_emb.unsqueeze(1)).sum(dim=0, keepdim=False))

        # aggregate and lambda
        former_node_emb = s_node_emb
        s_node_emb = self.the_GRU(s_nodes, s_node_emb, neighborhood_inf, community_inf)

        p_lambda = ((s_node_emb - t_node_emb) ** 2).sum(dim=1).neg()
        n_lambda = ((s_node_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg()
        w_lambda = torch.max(community_weight, dim=0, keepdim=True)[0].squeeze(0)

        # update community emb
        emb_diff = s_node_emb - former_node_emb
        weight_index = torch.max(community_weight, dim=0, keepdim=False)[1]
        for i in range(batch):
            community_emb[weight_index[i]] += emb_diff[i]

        self.node_emb[s_nodes] = s_node_emb.data
        self.com_emb = community_emb.data

        return p_lambda, n_lambda, w_lambda

    def loss_func(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                p_lambda, n_lambda, w_lambda = self.forward(s_nodes, t_nodes, t_times,
                                                            n_nodes, h_nodes, h_times, h_time_mask)
                loss = -(torch.log(p_lambda.sigmoid() + 1e-6)
                         + torch.log(n_lambda.neg().sigmoid() + 1e-6).sum(dim=1)
                         + torch.log(w_lambda.sigmoid() + 1e-6))
        else:
            p_lambda, n_lambda, w_lambda = self.forward(s_nodes, t_nodes, t_times,
                                                        n_nodes, h_nodes, h_times, h_time_mask)
            loss = -(torch.log(torch.sigmoid(p_lambda) + 1e-6)
                     + torch.log(torch.sigmoid(torch.neg(n_lambda)) + 1e-6).sum(dim=1)
                     + torch.log(torch.sigmoid(w_lambda) + 1e-6))
        return loss

    def update(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                self.opt.zero_grad()
                loss = self.loss_func(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
                loss = loss.sum()
                self.loss += loss.data
                loss.backward()
                self.opt.step()

        else:
            self.opt.zero_grad()
            loss = self.loss_func(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
            loss = loss.sum()
            self.loss += loss.data
            loss.backward()
            self.opt.step()

    def train(self):
        print('Training......')
        for epoch in range(self.epochs):
            once_start = datetime.datetime.now()
            self.loss = 0.0
            loader = DataLoader(self.data, batch_size=self.batch, shuffle=True, num_workers=4)

            if epoch % self.save_step == 0 and epoch != 0:
                self.save_node_embeddings(self.emb_path % (self.network, self.network, epoch))

            for i_batch, sample_batched in enumerate(loader):
                if i_batch != 0:
                    sys.stdout.write('\r' + str(i_batch * self.batch) + '\tloss: ' + str(
                        self.loss.cpu().numpy() / (self.batch * i_batch)))
                    sys.stdout.flush()

                if torch.cuda.is_available():
                    with torch.cuda.device(DID):
                        self.update(sample_batched['source_node'].type(LType).cuda(),
                                    sample_batched['target_node'].type(LType).cuda(),
                                    sample_batched['target_time'].type(FType).cuda(),
                                    sample_batched['neg_nodes'].type(LType).cuda(),
                                    sample_batched['history_nodes'].type(LType).cuda(),
                                    sample_batched['history_times'].type(FType).cuda(),
                                    sample_batched['history_masks'].type(FType).cuda())

                else:
                    self.update(sample_batched['source_node'].type(LType),
                                sample_batched['target_node'].type(LType),
                                sample_batched['target_time'].type(FType),
                                sample_batched['neg_nodes'].type(LType),
                                sample_batched['history_nodes'].type(LType),
                                sample_batched['history_times'].type(FType),
                                sample_batched['history_masks'].type(FType))

            once_end = datetime.datetime.now()
            sys.stdout.write('\repoch ' + str(epoch) + ': avg loss = ' + str(self.loss.cpu().numpy() / len(self.data))
                             + '\tonce_runtime: ' + str(once_end - once_start) + '\n')
            sys.stdout.flush()
            self.scheduler.step()

        self.save_node_embeddings(self.emb_path % (self.network, self.network, self.epochs))

    def save_node_embeddings(self, path):
        if torch.cuda.is_available():
            embeddings = self.node_emb.cpu().data.numpy()
        else:
            embeddings = self.node_emb.data.numpy()
        writer = open(path, 'w')
        writer.write('%d %d\n' % (self.node_dim, self.emb_size))
        for n_idx in range(self.node_dim):
            writer.write(str(n_idx) + ' ' + ' '.join(str(d) for d in embeddings[n_idx]) + '\n')

        writer.close()


if __name__ == '__main__':
    start = datetime.datetime.now()
    MNCI = MNCI()
    MNCI.train()
    end = datetime.datetime.now()
    print('total runtime: %s' % str(end - start))
