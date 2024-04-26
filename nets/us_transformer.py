# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import math
from typing import Tuple, Type

from .slimmable_ops import USLayerNorm, USLinear, make_divisible
max_width = 1.0

def transmitting_matrix(fm1: torch.Tensor, fm2: torch.Tensor):
    # fm1: B x C x H x W
    fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
    fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1, 2)

    fsp = torch.bmm(fm1, fm2) / fm1.size(2)
    return fsp

def top_eigenvalue(K: torch.Tensor, 
                   n_power_iterations: int = 10, 
                   dim: int = 1):
    v = torch.ones(K.shape[0], K.shape[1], 1).to(K.device)
    for _ in range(n_power_iterations):
        m = torch.bmm(K, v)
        n = torch.norm(m, dim=1).unsqueeze(1)
        v = m / n

    top_eigenvalue = torch.sqrt(n / torch.norm(v, dim=1).unsqueeze(1))
    return top_eigenvalue

def lipschitz_loss(s_in, s_out, t_in, t_out):
    TM_s = torch.bmm(transmitting_matrix(s_in, s_out), transmitting_matrix(s_in, s_out).transpose(2, 1))
    TM_t = torch.bmm(transmitting_matrix(t_in.detach(), t_out.detach()), transmitting_matrix(t_in.detach(), t_out.detach()).transpose(2, 1))
    return F.mse_loss(top_eigenvalue(K=TM_s), top_eigenvalue(K=TM_t))


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, 
        )
        self.norm_final_attn = USLayerNorm(embedding_dim, max_width=max_width)
        
        self._features = {}
        self.register_forward_hook(self.save_inputs_hook)
    
    def save_inputs_hook(self):
        def fn(_, inputs, outputs):
            self._features["inputs"] = inputs
            self._features["outputs"] = outputs

        return fn

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys

    def reuse_feature(self, queries, keys, point_embedding, image_pe):
        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys

    def forward_lipschitz_loss(self):
        # Prepare queries
        image_embedding, image_pe, point_embedding = self._features["inputs"]
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        t_in_queries = point_embedding
        t_in_keys = image_embedding

        # Apply width multiplier
        # queries, keys, point_embedding, image_pe = self._features["inputs"]
        idx = make_divisible(t_in_queries.shape[2] * self.width_mult)
        s_in_queries, s_in_keys, point_embedding, image_pe = (t_in_queries[..., :idx], t_in_keys[..., :idx],
                                                    point_embedding[..., :idx], image_pe[..., :idx])

        s_out_queries, s_out_keys = self.reuse_feature(
            queries=s_in_queries, keys=s_in_keys,
            point_embedding=point_embedding, image_pe=image_pe
        )
        t_out_queries, t_out_keys = self._features["outputs"]
        
        loss_q = lipschitz_loss(s_in=s_in_queries, s_out=s_out_queries, t_in=t_in_queries, t_out=t_out_queries)
        loss_k = lipschitz_loss(s_in=s_in_keys, s_out=s_out_keys, t_in=t_in_keys, t_out=t_out_keys)
        
        return loss_q + loss_k, loss_q.item(), loss_k.item()


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = USLayerNorm(embedding_dim, max_width=max_width)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = USLayerNorm(embedding_dim, max_width=max_width)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = USLayerNorm(embedding_dim, max_width=max_width)

        self.norm4 = USLayerNorm(embedding_dim, max_width=max_width)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        max_width: int = 1.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = USLinear(embedding_dim, self.internal_dim, max_width=max_width)
        self.k_proj = USLinear(embedding_dim, self.internal_dim, max_width=max_width)
        self.v_proj = USLinear(embedding_dim, self.internal_dim, max_width=max_width)
        self.out_proj = USLinear(self.internal_dim, embedding_dim, max_width=max_width)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = USLinear(embedding_dim, mlp_dim, us=[True, False], max_width=max_width)
        self.lin2 = USLinear(mlp_dim, embedding_dim, us=[False, True], max_width=max_width)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))