import torch
import torch.nn as nn
import torch.nn.functional as F
# from tutel import moe as tutel_moe
from .MOELayer import MoE
from .MODLayer import DecoderLayer

__all__ = ['MOEModel_Light']


class MOEModel_Light(nn.Module):
    """
        MOE implementations:
            (1) with tutel, ref to "https://github.com/microsoft/tutel"
            (2) with "https://github.com/davidmrau/mixture-of-experts"
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.eval_type = self.model_params['eval_type']
        self.problem = self.model_params['problem']
        self.aux_loss, self.T = 0, 1.0

        self.encoder = MTL_Encoder(**model_params)
        # self.decoder = MTL_Decoder(**model_params)
        self.decoder = MTL_DeeperDecoder(**model_params)
        self.encoded_nodes = None  # shape: (batch, problem+1, EMBEDDING_DIM)
        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in model_params.keys() else model_params['device']

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        node_tw_start = reset_state.node_tw_start
        node_tw_end = reset_state.node_tw_end
        # shape: (batch, problem)
        node_xy_demand_tw = torch.cat((node_xy, node_demand[:, :, None], node_tw_start[:, :, None], node_tw_end[:, :, None]), dim=2)
        # shape: (batch, problem, 5)
        # prob_emb = reset_state.prob_emb
        # shape: (1, 5) - only for problem-level routing/gating

        self.encoded_nodes, moe_loss = self.encoder(depot_xy, node_xy_demand_tw)
        self.aux_loss = moe_loss
        # shape: (batch, problem+1, embedding)
        self.decoder.set_kv(self.encoded_nodes)

    def set_eval_type(self, eval_type):
        self.eval_type = eval_type

    def forward(self, state, selected=None):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long).to(self.device)
            prob = torch.ones(size=(batch_size, pomo_size))
            # probs = torch.ones(size=(batch_size, pomo_size, self.encoded_nodes.size(1)))
            # shape: (batch, pomo, problem_size+1)

            # # Use Averaged encoded nodes for decoder input_1
            # encoded_nodes_mean = self.encoded_nodes.mean(dim=1, keepdim=True)
            # # shape: (batch, 1, embedding)
            # self.decoder.set_q1(encoded_nodes_mean)

            # # Use encoded_depot for decoder input_2
            # encoded_first_node = self.encoded_nodes[:, [0], :]
            # # shape: (batch, 1, embedding)
            # self.decoder.set_q2(encoded_first_node)

        elif state.selected_count == 1:  # Second Move, POMO
            # selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, -1).to(self.device)
            selected = state.START_NODE
            prob = torch.ones(size=(batch_size, pomo_size))
            # probs = torch.ones(size=(batch_size, pomo_size, self.encoded_nodes.size(1)))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            attr = torch.cat((state.load[:, :, None], state.current_time[:, :, None], state.length[:, :, None], state.open[:, :, None]), dim=2)
            # shape: (batch, pomo, 4)
            probs, moe_loss = self.decoder(encoded_last_node, attr, ninf_mask=state.ninf_mask, T=self.T, step=state.selected_count)
            self.aux_loss += moe_loss
            # shape: (batch, pomo, problem+1)
            if selected is None:
                while True:
                    if self.training or self.eval_type == 'softmax':
                        try:
                            selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
                        except Exception as exception:
                            print(">> Catch Exception: {}, on the instances of {}".format(exception, state.PROBLEM))
                            exit(0)
                    else:
                        selected = probs.argmax(dim=2)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break
            else:
                selected = selected
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class MTL_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        hidden_dim = self.model_params['ff_hidden_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        # [Option 1]: Use MoEs in Raw Features
        if self.model_params['num_experts'] > 1 and "Raw" in self.model_params['expert_loc']:
            self.embedding_depot = MoE(input_size=2, output_size=embedding_dim, num_experts=self.model_params['num_experts'],
                                       k=self.model_params['topk'], T=1.0, noisy_gating=True, routing_level=self.model_params['routing_level'],
                                       routing_method=self.model_params['routing_method'], moe_model="Linear")
            self.embedding_node = MoE(input_size=5, output_size=embedding_dim, num_experts=self.model_params['num_experts'],
                                      k=self.model_params['topk'], T=1.0, noisy_gating=True, routing_level=self.model_params['routing_level'],
                                      routing_method=self.model_params['routing_method'], moe_model="Linear")
        else:
            self.embedding_depot = nn.Linear(2, embedding_dim)
            self.embedding_node = nn.Linear(5, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(i, **model_params) for i in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand_tw):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand_tw.shape: (batch, problem, 5)
        # prob_emb: (1, embedding)

        moe_loss = 0
        if isinstance(self.embedding_depot, MoE) or isinstance(self.embedding_node, MoE):
            embedded_depot, loss_depot = self.embedding_depot(depot_xy)
            embedded_node, loss_node = self.embedding_node(node_xy_demand_tw)
            moe_loss = moe_loss + loss_depot + loss_node
        else:
            embedded_depot = self.embedding_depot(depot_xy)
            # shape: (batch, 1, embedding)
            embedded_node = self.embedding_node(node_xy_demand_tw)
            # shape: (batch, problem, embedding)

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape: (batch, problem+1, embedding)

        for layer in self.layers:
            out, loss = layer(out)
            moe_loss = moe_loss + loss

        return out, moe_loss
        # shape: (batch, problem+1, embedding)


class EncoderLayer(nn.Module):
    def __init__(self, depth=0, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        # [Option 2]: Use MoEs in Encoder
        if self.model_params['num_experts'] > 1 and "Enc{}".format(depth) in self.model_params['expert_loc']:
            # TODO: enabling parallelism
            # (1) MOE with tutel, ref to "https://github.com/microsoft/tutel"
            """
            assert self.model_params['routing_level'] == "node", "Tutel only supports node-level routing!"
            self.feedForward = tutel_moe.moe_layer(
                gate_type={'type': 'top', 'k': self.model_params['topk']},
                model_dim=embedding_dim,
                experts={'type': 'ffn', 'count_per_node': self.model_params['num_experts'],
                         'hidden_size_per_expert': self.model_params['ff_hidden_dim'],
                         'activation_fn': lambda x: F.relu(x)},
            )
            """
            # (2) MOE with "https://github.com/davidmrau/mixture-of-experts"
            self.feedForward = MoE(input_size=embedding_dim, output_size=embedding_dim, num_experts=self.model_params['num_experts'],
                                   hidden_size=self.model_params['ff_hidden_dim'], k=self.model_params['topk'], T=1.0, noisy_gating=True,
                                   routing_level=self.model_params['routing_level'], routing_method=self.model_params['routing_method'], moe_model="MLP")
        else:
            self.feedForward = FeedForward(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        """
        Two implementations:
            norm_last: the original implementation of AM/POMO: MHA -> Add & Norm -> FFN/MOE -> Add & Norm
            norm_first: the convention in NLP: Norm -> MHA -> Add -> Norm -> FFN/MOE -> Add
        """
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num, moe_loss = self.model_params['head_num'], 0

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        if self.model_params['norm_loc'] == "norm_last":
            out_concat = multi_head_attention(q, k, v)  # (batch, problem, HEAD_NUM*KEY_DIM)
            multi_head_out = self.multi_head_combine(out_concat)  # (batch, problem, EMBEDDING_DIM)
            out1 = self.addAndNormalization1(input1, multi_head_out)
            out2, moe_loss = self.feedForward(out1)
            out3 = self.addAndNormalization2(out1, out2)  # (batch, problem, EMBEDDING_DIM)
        else:
            out1 = self.addAndNormalization1(None, input1)
            multi_head_out = self.multi_head_combine(out1)
            input2 = input1 + multi_head_out
            out2 = self.addAndNormalization2(None, input2)
            out2, moe_loss = self.feedForward(out2)
            out3 = input2 + out2

        return out3, moe_loss


########################################
# DECODER
########################################

class MTL_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        self.hierarchical_gating = False

        # self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim + 4, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        # [Option 3]: Use MoEs in Decoder
        if self.model_params['num_experts'] > 1 and 'Dec' in self.model_params['expert_loc']:
            self.hierarchical_gating = True
            self.dense_or_moe = nn.Linear(head_num * qkv_dim, 2, bias=False)
            self.multi_head_combine_moe = MoE(input_size=head_num * qkv_dim, output_size=embedding_dim, num_experts=self.model_params['num_experts'],
                                              k=self.model_params['topk'], T=1.0, noisy_gating=True, routing_level=self.model_params['routing_level'],
                                              routing_method=self.model_params['routing_method'], moe_model="Linear")
            self.multi_head_combine_dense = nn.Linear(head_num * qkv_dim, embedding_dim)
        else:
            self.multi_head_combine_dense = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        # self.q1 = None  # saved q1, for multi-head attention
        # self.q2 = None  # saved q2, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q2 = reshape_by_heads(self.Wq_2(encoded_q2), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, attr, ninf_mask, T=1.0, step=2):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # attr.shape: (batch, pomo, 4)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num, moe_loss = self.model_params['head_num'], 0

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, attr), dim=2)
        # shape = (batch, group, EMBEDDING_DIM + 4)

        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        # q = self.q1 + self.q2 + q_last
        # # shape: (batch, head_num, pomo, qkv_dim)
        q = q_last
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        if self.hierarchical_gating:
            if step == 2:  # this line could be removed if using Gating_Network_1: dense_or_moe in every decoding step
                self.probs = F.softmax(self.dense_or_moe(out_concat.mean(0).mean(0).unsqueeze(0)) / T, dim=-1)  # [1, 2]
            selected = self.probs.multinomial(1).squeeze(0)
            if selected.item() == 1:
                mh_atten_out, moe_loss = self.multi_head_combine_moe(out_concat)
            else:
                mh_atten_out = self.multi_head_combine_dense(out_concat)
            mh_atten_out = mh_atten_out * self.probs.squeeze(0)[selected]
        else:
            mh_atten_out = self.multi_head_combine_dense(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs, moe_loss

class MTL_DeeperDecoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        self.hierarchical_gating = False

        # self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim + 4, head_num * qkv_dim, bias=False)
        # self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.decoder_layers = nn.ModuleList([DecoderLayer(**model_params) for i in range(self.model_params['decoder_layer_num'])])

        # [Option 3]: Use MoEs in Decoder
        if self.model_params['num_experts'] > 1 and 'Dec' in self.model_params['expert_loc']:
            self.hierarchical_gating = True
            self.dense_or_moe = nn.Linear(head_num * qkv_dim, 2, bias=False)
            self.multi_head_combine_moe = MoE(input_size=head_num * qkv_dim, output_size=embedding_dim, num_experts=self.model_params['num_experts'],
                                              k=self.model_params['topk'], T=1.0, noisy_gating=True, routing_level=self.model_params['routing_level'],
                                              routing_method=self.model_params['routing_method'], moe_model="Linear")
            self.multi_head_combine_dense = nn.Linear(head_num * qkv_dim, embedding_dim)
        else:
            self.multi_head_combine_dense = nn.Linear(head_num * qkv_dim, embedding_dim)

        # self.k = None  # saved key, for multi-head attention
        # self.v = None  # saved value, for multi-head_attention
        self.h = None
        self.single_head_key = None  # saved, for single-head attention
        # self.q1 = None  # saved q1, for multi-head attention
        # self.q2 = None  # saved q2, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.h = encoded_nodes
        # self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        # self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q2 = reshape_by_heads(self.Wq_2(encoded_q2), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, attr, ninf_mask, T=1.0, step=2):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # attr.shape: (batch, pomo, 4)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num, moe_loss = self.model_params['head_num'], 0

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, attr), dim=2)
        # shape = (batch, group, EMBEDDING_DIM + 4)

        # q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        q_last =self.Wq_last(input_cat)
        # shape: (batch, head_num, pomo, qkv_dim)

        # q = self.q1 + self.q2 + q_last
        # # shape: (batch, head_num, pomo, qkv_dim)
        q = q_last
        # shape: (batch, head_num, pomo, qkv_dim)

        h = self.h
        for n, layer in enumerate(self.decoder_layers):
            q, _, h = layer(q, h, h, rank3_ninf_mask=ninf_mask)
        # out_concat = multi_head_attention(q, self.k, self.v,
        out_concat = q
        # rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        if self.hierarchical_gating:
            if step == 2:  # this line could be removed if using Gating_Network_1: dense_or_moe in every decoding step
                self.probs = F.softmax(self.dense_or_moe(out_concat.mean(0).mean(0).unsqueeze(0)) / T, dim=-1)  # [1, 2]
            selected = self.probs.multinomial(1).squeeze(0)
            if selected.item() == 1:
                mh_atten_out, moe_loss = self.multi_head_combine_moe(out_concat)
            else:
                mh_atten_out = self.multi_head_combine_dense(out_concat)
            mh_atten_out = mh_atten_out * self.probs.squeeze(0)[selected]
        else:
            mh_atten_out = self.multi_head_combine_dense(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs, moe_loss
########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.add = True if 'norm_loc' in model_params.keys() and model_params['norm_loc'] == "norm_last" else False
        if model_params["norm"] == "batch":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=True)
        elif model_params["norm"] == "batch_no_track":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "instance":
            self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "layer":
            self.norm = nn.LayerNorm(embedding_dim)
        elif model_params["norm"] == "rezero":
            self.norm = torch.nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        else:
            self.norm = None

    def forward(self, input1=None, input2=None):
        # input.shape: (batch, problem, embedding)
        if isinstance(self.norm, nn.InstanceNorm1d):
            added = input1 + input2 if self.add else input2
            transposed = added.transpose(1, 2)
            # shape: (batch, embedding, problem)
            normalized = self.norm(transposed)
            # shape: (batch, embedding, problem)
            back_trans = normalized.transpose(1, 2)
            # shape: (batch, problem, embedding)
        elif isinstance(self.norm, nn.BatchNorm1d):
            added = input1 + input2 if self.add else input2
            batch, problem, embedding = added.size()
            normalized = self.norm(added.reshape(batch * problem, embedding))
            back_trans = normalized.reshape(batch, problem, embedding)
        elif isinstance(self.norm, nn.LayerNorm):
            added = input1 + input2 if self.add else input2
            back_trans = self.norm(added)
        elif isinstance(self.norm, nn.Parameter):
            back_trans = input1 + self.norm * input2 if self.add else self.norm * input2
        else:
            back_trans = input1 + input2 if self.add else input2

        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1))), 0
