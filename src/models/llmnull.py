


import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, compute_normalized_laplacian


class LLMNULL(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LLMNULL, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']   # 总维度
        self.knn_k = config['knn_k']     # KNN图的近邻数
        self.n_layers = config['n_mm_layers']   # 多模态交互层数
        self.n_ui_layers = config['n_ui_layers']   # 用户-物品交互层数
        self.reg_weight = config['reg_weight']   # 权重
        self.build_item_graph = True
        self.device = config['device'] 
        self.mm_image_weight = config['mm_image_weight']   # 图像模态权重
        self.dropout = config['dropout']


        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        # 用户-物品交互矩阵
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)     # 归一化邻接矩阵，用于整体图的计算
        self.masked_adj, self.mm_adj = None, None
         # 边信息，edge_indices和edge_values存储图的连接关系和权重
        self.edge_indices, self.edge_values = self.get_edge_info()   # 用户-物品交互矩阵中提取边信息，用于边级别的操作
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)    # 每条边分配一个唯一的索引


        # 嵌入层初始化
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # 添加AlphaFuse模块配置
        self.use_alphafuse = config['use_alphafuse']
        if self.use_alphafuse:
            self.alphafuse_embedder = AlphaFuseItemEmbedder(
                v_feat=self.v_feat,
                t_feat=self.t_feat,
                device =self.device,
                embedding_dim=self.embedding_dim,
                null_thres=config['null_thres'],
                null_dim=config['null_dim'],
                standardization=config['standardization'],
                cover=config['cover'],
                ID_space=config['ID_space'],
                inject_space=config['inject_space'],
                emb_init_type=config['emb_init_type'],
                emb_type=config['emb_type']
            )

        # 构建或加载多模态邻接矩阵
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_freedomdsp_{}_{}.pt'.format(self.knn_k, int(10*self.mm_image_weight)))

        # 处理和利用多模态特征，将它们转换为模型可使用的嵌入表示。
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)   # 将预计算的图像特征转换为 PyTorch 的嵌入层，使得模型可以微调这些特征
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)


        

        
        # 构建多模态图
        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())   # 通过get_knn_adj_mat()基于特征相似度构建物品间的邻接矩阵（基于原始图像嵌入）
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj   # 若同时存在图像和文本特征，通过mm_image_weight加权融合两种模态的邻接矩阵
                # self.mm_adj = image_adj + text_adj   # 若同时存在图像和文本特征，通过mm_image_weight加权融合两种模态的邻接矩阵
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)

    # 实现了基于 K 近邻（KNN）算法构建图的邻接矩阵，仅保留每个节点的 K 个最相似邻居，self.knn_k
    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    # 计算图神经网络中的归一化拉普拉斯矩阵，即计算并返回图的对称归一化邻接矩阵  
    # 更通用，更常用于物品图
    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)   # 每个非零元素初始化为 1，表示有连接
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()    # # 计算每个节点的度
        r_inv_sqrt = torch.pow(row_sum, -0.5)    # 计算度的平方根的倒数
        rows_inv_sqrt = r_inv_sqrt[indices[0]]    # 节点i的度的平方根的倒数
        cols_inv_sqrt = r_inv_sqrt[indices[1]]     # 节点j的度的平方根的倒数

        values = rows_inv_sqrt * cols_inv_sqrt    #  # 边权重 = 1/(sqrt(d[i])*sqrt(d[j]))  归一化，平衡了不同度的节点在网络中的影响力
        return torch.sparse.FloatTensor(indices, values, adj_size)

    # 构建并归一化用户 - 物品二分图的邻接矩阵，返回的是邻接矩阵的稀疏张量，用户消息传递
    # 针对用户-物品交互图进行设计，用于交互图
    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)    # (n_users + n_items) × (n_users + n_items) 的方阵
        inter_M = self.interaction_matrix    # 用户 - 项目交互矩阵
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),    # 用户→项目连接,将inter_M中非零元素的坐标映射到邻接矩阵A的右上角
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),    # 项目→用户连接,将inter_M_t中非零元素的坐标映射到邻接矩阵A的左下角
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix   对称归一化处理
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D      # 计算对称归一化后的矩阵 L = D^(-1/2) * A * D^(-1/2)，同上，也是权重除度
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    # 在每次训练 epoch 前，根据设定的 dropout 率随机删除部分用户 - 物品交互边，构建一个随机掩码的邻接矩阵 masked_adj
    # 度数敏感边剪枝:根据边的重要性权重（基于节点度数计算）进行选择性剪枝
    def pre_epoch_processing(self):
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            return
        # degree-sensitive edge pruning  基于度数计算边保留概率
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
        # 根据概率向量self.edge_values采样degree_len条边，实现度数敏感的边剪枝
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        # random sample  存储采样后保留的边的坐标
        keep_indices = self.edge_indices[:, degree_idx]
        # norm values  重新归一化保留的边
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    # 也是在计算图的对称归一化邻接矩阵的一部分，但只返回计算后的 values（归一化后的边权重）
    # 输入是边索引和矩阵大小，动态构建子图（如边 dropout 后的图）；get_norm_adj_mat输入和输出的是完整的矩阵
    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()    # 用户度
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()   # 物品度
        # 为每条边计算归一化权重：1/sqrt(用户度 * 物品度)
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    # 从用户 - 物品交互矩阵中提取边信息并进行归一化
    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)   
        cols = torch.from_numpy(self.interaction_matrix.col)  
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values


    #-----------11111111111:原始-0.0624-0.0978-----------------------
    # def forward(self, adj):
    #     # 多模态图计算物品表征（使用初始化的嵌入）
    #     h = self.item_id_embedding.weight
    #     for i in range(self.n_layers):
    #         h = torch.sparse.mm(self.mm_adj, h)

    #     # 交互图计算用户和物品表征（使用初始化的嵌入）
    #     ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
    #     all_embeddings = [ego_embeddings]
    #     for i in range(self.n_ui_layers):
    #         side_embeddings = torch.sparse.mm(adj, ego_embeddings)
    #         ego_embeddings = side_embeddings
    #         all_embeddings += [ego_embeddings]
    #     all_embeddings = torch.stack(all_embeddings, dim=1)
    #     all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
    #     u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
    #     return u_g_embeddings, i_g_embeddings + h



    # # #-----------2222222222222：不初始化物品嵌入-0.0635-0.0991----------------------
    # # #-----------2222222222222：文本和图像模态权重均为1-0.0624-0.0978----------------------
    # def forward(self, adj):
    #     # 多模态图计算物品表征（使用初始化的嵌入）
    #     h = self.item_id_embedding.weight
    #     for i in range(self.n_layers):
    #         h = torch.sparse.mm(self.mm_adj, h)

    #     # 交互图计算用户和物品表征（使用初始化的嵌入）
    #     ego_embeddings = torch.cat((self.user_embedding.weight, h), dim=0)
    #     all_embeddings = [ego_embeddings]
    #     for i in range(self.n_ui_layers):
    #         side_embeddings = torch.sparse.mm(adj, ego_embeddings)
    #         ego_embeddings = side_embeddings
    #         all_embeddings += [ego_embeddings]
    #     all_embeddings = torch.stack(all_embeddings, dim=1)
    #     all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
    #     u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
    #     return u_g_embeddings, i_g_embeddings

    #-----------33333333333------------------------
    def forward(self, adj):
        if self.use_alphafuse:
            # 使用AlphaFuse进行物品嵌入
            item_ids = torch.arange(self.n_items).to(self.device)
            alphafuse_item_embs = self.alphafuse_embedder(item_ids)  # 使用融合嵌入
            
            # # 多模态图传播（使用AlphaFuse嵌入）
            # h = alphafuse_item_embs
            # for i in range(self.n_layers):
            #     h = torch.sparse.mm(self.mm_adj, h)
            
            # 交互图计算用户和物品表征
            ego_embeddings = torch.cat((self.user_embedding.weight, alphafuse_item_embs), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            
            return u_g_embeddings, i_g_embeddings
        else:
            # 原始方法（不使用AlphaFuse）
            h = self.item_id_embedding.weight
            for i in range(self.n_layers):
                h = torch.sparse.mm(self.mm_adj, h)

            # 交互图计算用户和物品表征（使用初始化的嵌入）
            ego_embeddings = torch.cat((self.user_embedding.weight, h), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            return u_g_embeddings, i_g_embeddings




    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)    # 计算用户与正样本的交互分数
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)    # 计算用户与负样本的交互分数

        maxi = F.logsigmoid(pos_scores - neg_scores)  # 计算正样本分数与负样本分数的差值，并通过sigmoid和log转换
        mf_loss = -torch.mean(maxi)

        return mf_loss

    # 基础 BPR 损失 + 文本模态损失 + 图像模态损失
    def calculate_loss(self, interaction):
        # 三元组
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings = self.forward(self.masked_adj)
        self.build_item_graph = False

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        # 基础 BPR 损失
        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,neg_i_g_embeddings)
        
        return batch_mf_loss 

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores



# Cross-modal attention mechanism for enhanced fusion
class CrossModalAttention(nn.Module):
    def __init__(self, modal_dim, attention_dim=None):
        super(CrossModalAttention, self).__init__()
        if attention_dim is None:
            attention_dim = modal_dim
        
        self.modal_dim = modal_dim
        self.attention_dim = attention_dim
        
        # Query, Key, Value projections
        self.query_proj = nn.Linear(modal_dim, attention_dim)
        self.key_proj = nn.Linear(modal_dim, attention_dim)
        self.value_proj = nn.Linear(modal_dim, attention_dim)
        
        # Output projection to maintain dimension
        self.output_proj = nn.Linear(attention_dim, modal_dim)
        
        # Scale factor for attention
        self.scale = attention_dim ** -0.5
        
    def forward(self, visual_emb, text_emb):
        """
        Compute cross-modal attention between visual and text embeddings
        
        Args:
            visual_emb: [batch_size, modal_dim] visual embeddings
            text_emb: [batch_size, modal_dim] text embeddings
            
        Returns:
            v2t_enhanced: visual embeddings enhanced by text information
            t2v_enhanced: text embeddings enhanced by visual information
        """
        batch_size = visual_emb.size(0)
        
        # Visual-to-Text attention
        v_query = self.query_proj(visual_emb)  # [batch_size, attention_dim]
        t_key = self.key_proj(text_emb)        # [batch_size, attention_dim]
        t_value = self.value_proj(text_emb)    # [batch_size, attention_dim]
        
        # Compute attention scores
        v2t_scores = torch.matmul(v_query.unsqueeze(1), t_key.unsqueeze(2)) * self.scale  # [batch_size, 1, 1]
        v2t_attention = torch.softmax(v2t_scores.squeeze(-1), dim=-1)  # [batch_size, 1]
        v2t_attended = torch.matmul(v2t_attention.unsqueeze(1), t_value.unsqueeze(1)).squeeze(1)  # [batch_size, attention_dim]
        v2t_enhanced = visual_emb + self.output_proj(v2t_attended)
        
        # Text-to-Visual attention
        t_query = self.query_proj(text_emb)    # [batch_size, attention_dim]
        v_key = self.key_proj(visual_emb)      # [batch_size, attention_dim]
        v_value = self.value_proj(visual_emb)  # [batch_size, attention_dim]
        
        # Compute attention scores
        t2v_scores = torch.matmul(t_query.unsqueeze(1), v_key.unsqueeze(2)) * self.scale  # [batch_size, 1, 1]
        t2v_attention = torch.softmax(t2v_scores.squeeze(-1), dim=-1)  # [batch_size, 1]
        t2v_attended = torch.matmul(t2v_attention.unsqueeze(1), v_value.unsqueeze(1)).squeeze(1)  # [batch_size, attention_dim]
        t2v_enhanced = text_emb + self.output_proj(t2v_attended)
        
        return v2t_enhanced, t2v_enhanced


# Unified null space for fused multi-modal embeddings
class UnifiedNullSpace(nn.Module):
    def __init__(self, embedding_dim, null_dim, standardization=True):
        super(UnifiedNullSpace, self).__init__()
        self.embedding_dim = embedding_dim
        self.null_dim = null_dim
        self.standardization = standardization
        
    def inject(self, fused_semantic, unified_id_emb, cover=False):
        """
        Inject unified ID embeddings into the fused semantic space
        
        Args:
            fused_semantic: [batch_size, embedding_dim] fused semantic embeddings
            unified_id_emb: [batch_size, null_dim] unified ID embeddings
            cover: whether to cover or add to the null space
            
        Returns:
            final_embedding: [batch_size, embedding_dim] final fused embeddings
        """
        result = fused_semantic.clone()
        
        if cover:
            # Cover mode: replace the null space with ID embeddings
            result[..., -self.null_dim:] = unified_id_emb
        else:
            # Add mode: add ID embeddings to the null space
            result[..., -self.null_dim:] = fused_semantic[..., -self.null_dim:] + unified_id_emb
            
        return result


# Improved multi-modal AlphaFuse module
class MultiModalAlphaFuse(nn.Module):
    def __init__(self, v_feat, t_feat, device, embedding_dim, null_thres, null_dim,
                 standardization, cover, ID_space, inject_space, emb_init_type, emb_type):
        super(MultiModalAlphaFuse, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.null_dim = null_dim
        self.standardization = standardization
        self.cover = cover
        self.ID_space = ID_space
        self.inject_space = inject_space
        self.emb_init_type = emb_init_type
        self.emb_type = emb_type
        self.device = device
        
        # Modal dimension for each modality (half of total embedding dim)
        self.modal_dim = embedding_dim // 2
        
        # Feature storage
        self.v_feat = v_feat
        self.t_feat = t_feat
        
        # Build modal projectors and semantic embeddings
        self._build_modal_spaces()
        
        # Cross-modal attention mechanism
        if self.v_feat is not None and self.t_feat is not None:
            self.cross_modal_attention = CrossModalAttention(self.modal_dim)
            # Fusion layer to combine enhanced modalities
            self.fusion_layer = nn.Sequential(
                nn.Linear(2 * self.modal_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )
        else:
            # If only one modality, use projection to full dimension
            self.single_modal_proj = nn.Linear(self.modal_dim, embedding_dim)
        
        # Unified null space and ID embeddings
        self.unified_null_space = UnifiedNullSpace(embedding_dim, null_dim, standardization)
        n_items = v_feat.size(0) if v_feat is not None else t_feat.size(0)
        self.unified_id_embeddings = nn.Embedding(n_items, null_dim)
        self._init_embedding(self.unified_id_embeddings, emb_init_type)
        
    def _build_modal_spaces(self):
        """Build semantic embeddings for each modality"""
        if self.v_feat is not None:
            v_feat_np = self.v_feat.detach().cpu().numpy()
            self._construct_modal_space(v_feat_np, "visual", self.modal_dim)
        
        if self.t_feat is not None:
            t_feat_np = self.t_feat.detach().cpu().numpy()
            self._construct_modal_space(t_feat_np, "text", self.modal_dim)
    
    def _construct_modal_space(self, feat_matrix, modal_name, target_dim):
        """Construct semantic space for a modality"""
        print(f"构建{modal_name}模态空间分解...")
        mean = np.mean(feat_matrix, axis=0)
        cov = np.cov(feat_matrix - mean, rowvar=False)
        U, S, _ = np.linalg.svd(cov, full_matrices=False)
        
        if self.standardization:
            P = U.dot(np.diag(np.sqrt(1/S)))
        else:
            P = U
        
        # Project features to target dimension
        projected_feat = (feat_matrix - mean).dot(P[:, :target_dim])
        
        # Create frozen semantic embedding
        semantic_emb = nn.Embedding.from_pretrained(
            torch.tensor(projected_feat, dtype=torch.float32),
            freeze=True
        )
        setattr(self, f'{modal_name}_semantic_embeddings', semantic_emb)
        print(f"{modal_name}模态语义空间维度: {target_dim}")
    
    def _init_embedding(self, embedding_layer, emb_init_type):
        """Initialize embedding layers"""
        if emb_init_type == "uniform":
            nn.init.uniform_(embedding_layer.weight, a=0.0, b=1.0)
        elif emb_init_type == "normal":
            nn.init.normal_(embedding_layer.weight, 0, 1)
        elif emb_init_type == "zeros":
            nn.init.zeros_(embedding_layer.weight)
        elif emb_init_type == "ortho":
            nn.init.orthogonal_(embedding_layer.weight, gain=1.0)
        elif emb_init_type == "xavier":
            nn.init.xavier_uniform_(embedding_layer.weight, gain=1.0)
        elif emb_init_type == "sparse":
            nn.init.sparse_(embedding_layer.weight, 0.01, std=1)
        else:
            nn.init.xavier_uniform_(embedding_layer.weight, gain=1.0)
    
    def forward(self, item_ids):
        """
        Forward pass for improved multi-modal fusion
        
        Args:
            item_ids: [batch_size] item indices
            
        Returns:
            final_embedding: [batch_size, embedding_dim] fused embeddings
        """
        # Get semantic embeddings for each modality
        modal_embeddings = []
        
        if self.v_feat is not None:
            visual_semantic = self.visual_semantic_embeddings(item_ids)
            modal_embeddings.append(visual_semantic)
        
        if self.t_feat is not None:
            text_semantic = self.text_semantic_embeddings(item_ids)
            modal_embeddings.append(text_semantic)
        
        # Handle different scenarios based on available modalities
        if len(modal_embeddings) == 2:
            # Both visual and text modalities available
            visual_semantic, text_semantic = modal_embeddings
            
            if self.emb_type == "semantic":
                # Only semantic information, no ID injection
                # Apply cross-modal attention
                enhanced_visual, enhanced_text = self.cross_modal_attention(visual_semantic, text_semantic)
                
                # Fuse enhanced modalities
                fused_input = torch.cat([enhanced_visual, enhanced_text], dim=-1)
                fused_semantic = self.fusion_layer(fused_input)
                
                return fused_semantic
            
            elif self.emb_type == "id":
                # Only ID embeddings, no semantic information
                unified_id_emb = self.unified_id_embeddings(item_ids)
                # Create zero semantic base and inject ID
                zero_semantic = torch.zeros(item_ids.size(0), self.embedding_dim, 
                                          device=item_ids.device, dtype=torch.float32)
                return self.unified_null_space.inject(zero_semantic, unified_id_emb, cover=True)
            
            else:  # "both" - semantic + ID
                # Apply cross-modal attention to enhance semantic representations
                enhanced_visual, enhanced_text = self.cross_modal_attention(visual_semantic, text_semantic)
                
                # Fuse enhanced modalities
                fused_input = torch.cat([enhanced_visual, enhanced_text], dim=-1)
                fused_semantic = self.fusion_layer(fused_input)
                
                # Inject unified ID embeddings into the fused space
                unified_id_emb = self.unified_id_embeddings(item_ids)
                final_embedding = self.unified_null_space.inject(fused_semantic, unified_id_emb, self.cover)
                
                return final_embedding
        
        elif len(modal_embeddings) == 1:
            # Only one modality available
            single_semantic = modal_embeddings[0]
            
            if self.emb_type == "semantic":
                # Project single modality to full dimension
                return self.single_modal_proj(single_semantic)
            
            elif self.emb_type == "id":
                # Only ID embeddings
                unified_id_emb = self.unified_id_embeddings(item_ids)
                zero_semantic = torch.zeros(item_ids.size(0), self.embedding_dim,
                                          device=item_ids.device, dtype=torch.float32)
                return self.unified_null_space.inject(zero_semantic, unified_id_emb, cover=True)
            
            else:  # "both"
                # Project to full dimension and inject ID
                projected_semantic = self.single_modal_proj(single_semantic)
                unified_id_emb = self.unified_id_embeddings(item_ids)
                final_embedding = self.unified_null_space.inject(projected_semantic, unified_id_emb, self.cover)
                
                return final_embedding
        
        else:
            # No modalities available - should not happen in practice
            raise ValueError("No visual or text features available")


# 新增的 AlphaFuse 的物品嵌入模块 (保持向后兼容)
class AlphaFuseItemEmbedder(nn.Module):
    
    def __init__(self, v_feat, t_feat, device, embedding_dim, null_thres, null_dim, 
                 standardization, cover, ID_space, inject_space, emb_init_type, emb_type):
        super(AlphaFuseItemEmbedder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.v_feat = v_feat
        self.t_feat = t_feat
        self.device = device
        
        # Use improved multi-modal AlphaFuse
        self.multi_modal_fuser = MultiModalAlphaFuse(
            v_feat=v_feat,
            t_feat=t_feat,
            device=device,
            embedding_dim=embedding_dim,
            null_thres=null_thres,
            null_dim=null_dim,
            standardization=standardization,
            cover=cover,
            ID_space=ID_space,
            inject_space=inject_space,
            emb_init_type=emb_init_type,
            emb_type=emb_type
        )
    
    def forward(self, item_ids):
        """
        Forward pass using improved multi-modal fusion
        """
        return self.multi_modal_fuser(item_ids)