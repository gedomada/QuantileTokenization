import torch
import torch.nn as nn
from torch import Tensor, IntTensor, FloatTensor

class TokenMasking(nn.Module):
    def __init__(self, p:float=0.1, mask_value=0):
        super(TokenMasking, self).__init__()
        self.mask_value = mask_value
        self.p = p
        
    def forward(self, x: torch.IntTensor) -> torch.IntTensor:
        if self.training and self.p > 0:
            mask = torch.rand(x.shape, device=x.device) < self.p
            x = torch.where(mask, x, self.mask_value)
        return x
    
class Embedding(nn.Embedding):
    def __init__(
            self, num_embeddings, embedding_dim, 
            max_norm:float=None, padding_idx:int=None, 
            dropout:float=0., token_masking:float=0.,
            init:str='none', init_kwargs:dict={}, 
            layernorm:bool=False,
        ):
        self.norm = layernorm
        self.pad = padding_idx is None and token_masking > 0

        super(Embedding, self).__init__(num_embeddings+ self.pad, embedding_dim, max_norm=max_norm, padding_idx=padding_idx)
        assert init in ['none', 'kaiming_uniform', 'uniform', 'normal']

        self.dropout = nn.Dropout(dropout)
        self.masking = TokenMasking(p=token_masking, mask_value=0 if padding_idx is None else padding_idx)
    
        if layernorm:
            self.layernorm = nn.LayerNorm(embedding_dim)

        if init == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.weight, **init_kwargs)
        elif init == 'uniform':
            nn.init.uniform_(self.weight, **init_kwargs)
        elif init == 'normal':
            nn.init.normal_(self.weight, **init_kwargs)

    def forward(self, inputs: IntTensor) -> FloatTensor:
        # |inputs| : (*)
        outputs = self.masking(inputs + 1 if self.pad else inputs)
        outputs = super().forward(outputs)
        outputs = self.dropout(outputs)
        # |outputs| : (*, H), where H - embedding_dim

        if self.norm:
            outputs = self.layernorm(outputs)
        # |outputs| : (*, H), where H - embedding_dim

        return outputs
    
class QuantileTokenization(nn.Module):
    """
    A PyTorch module that quantizes continuous features by quantile buckets.

    To use this module, first fit it to the feature distribution of the entire training corpus
    by calling the `fit()` method. This works for any: 1D or 2D vector.
    """
    def __init__(
        self, q_num: int|list, f_num: int, embedding_dim: int, 
        mode='mean', init:str='none', init_kwargs:dict={}, 
        max_norm:float=None, token_masking:float=0, dropout:float=0,
        ):
        """    
        Args:
        - q_num (int, list): The number of quantiles or list of quantiles to use for bucketization.
        - f_num (int): The number of features in the input data.
        - embedding_dim (int): The dimension of the learned embedding.
        - mode (str): The method used to calculate the output representation. It can be one of
        'mean', 'sum', `none`, or 'flatten' (default is 'mean').

        Attributes:
        - boundaries (nn.Parameter): The learned boundaries between the quantiles.
        - emb (Embedding): The learned embedding.
        """
        super(QuantileTokenization, self).__init__()
        assert mode in ['none', 'mean', 'sum', 'flatten']
        
        if isinstance(q_num, list):
            self.custom_quantiles = True

            quantiles, q_num = q_num, len(q_num)
            for q in quantiles: assert 0 <= q <= 1
            quantiles.sort()

        else:
            self.custom_quantiles = False

            q_step = 1/q_num
            quantiles = torch.arange(q_step, 1+1/q_num, q_step).tolist()

        self.mode = mode
        self.q_num = q_num
        self.f_num = f_num
        self.quantiles: list[float] = quantiles
        self.embedding_dim = embedding_dim
        self.boundaries = nn.Parameter(torch.empty(f_num,  q_num), requires_grad=False)
        self.emb = Embedding(
            self.max_token_id + 1, embedding_dim, 
            init=init,  init_kwargs=init_kwargs, max_norm=max_norm, 
            token_masking=token_masking, dropout=dropout
        )

    @property
    def max_token_id(self) -> int:
        """
        Returns the maximum possible token ID based on the number of quantiles used for bucketization.
        +1 Cause we keep 0 token for masking, or smth
        """
        ts = self.q_num * self.f_num + 1
        return ts
    
    @property
    def out_dim(self) -> int:
        if self.mode == 'flatten':
            return self.embedding_dim * self.f_num
        else:
            return self.embedding_dim
    
    def fit(self, x: Tensor) -> 'QuantileTokenization':
        """
        Fits the module to the quantile distribution of `x`.

        Args:
            x (Tensor): The tensor to fit the module to.
            
        Returns:
            The fitted module.
        """
        assert x.ndim in (1, 2, 3)
        
        if x.ndim == 3:
            b, t, f = x.shape
            x = torch.reshape(x, (b*t, f))
        
        if x.ndim in (2, 3):
            assert x.size(-1) == self.f_num
            
        # Calculate boundaries using quantiles
        if x.dtype in [torch.int64, torch.int32]:
            boundaries = torch.quantile(x.double(), torch.tensor(self.quantiles, dtype=torch.double, device=x.device), dim=0).type_as(x)
        else:
            boundaries = torch.quantile(x, torch.tensor(self.quantiles, dtype=x.dtype, device=x.device), dim=0)
        
        if boundaries.ndim == 2:
            boundaries.swapaxes_(0, 1)
        if boundaries.ndim == 1:
            boundaries.unsqueeze_(0)

        self.boundaries = nn.Parameter(boundaries, requires_grad=False)
        # |boundaries| - (self.f_num, self.q_num)
        return self
    
    def bucketize(self, x: Tensor) -> IntTensor:
        assert x.ndim in (1, 2, 3)
        in_shape = x.shape

        if self.f_num == 1:
            _x = x.view(-1).unsqueeze(0).contiguous()
        elif x.ndim == 3:
            b, t, f = in_shape
            _x = torch.reshape(x, (b*t, f))
            _x = _x.swapaxes(0, 1).contiguous()
        elif x.ndim == 2:
            _x = x.swapaxes(0, 1).contiguous()
        else:
            _x = x
            
        # Quantize the input features using the learned boundaries
        if _x.ndim == 2:
            quantized = []
            for idx, (_f, q) in enumerate(zip(_x, self.boundaries)):
                buckets = torch.bucketize(_f, q)
                buckets = buckets + (idx*self.q_num)
                # Add 1 because 0 is reserved for padding/masking/missing token
                quantized.append(buckets + 1)

            quantized = torch.stack(quantized)
            quantized.swapaxes_(0, 1)
            
        else:
            assert self.f_num == 1
            buckets = torch.bucketize(_x, self.boundaries[0])
            quantized = self.emb(buckets)
        
        # Reshape to original shape
        if self.f_num == 1:
            quantized = quantized.view(*in_shape)

        if x.ndim == 3:
            quantized = torch.reshape(quantized, (b, t, f))
        
        return quantized
    
    def embed(self, x: IntTensor) -> FloatTensor:
        assert x.ndim in (1, 2, 3)

        embs = self.emb(x)
        
        if self.mode == 'mean':
            embs = embs.mean(dim=-2)
        elif self.mode == 'sum':
            embs = embs.sum(dim=-2)
        elif self.mode == 'flatten':
            embs = torch.flatten(embs, start_dim=-2, end_dim=-1)

        return embs
    
    def forward(self, x: FloatTensor) -> FloatTensor:
        assert x.ndim in (1, 2, 3)
    
        bucketized = self.bucketize(x)
        embs = self.embed(bucketized)
        return embs

    def extra_repr(self) -> str:
        q_repr = f"quantiles={self.quantiles}" if self.custom_quantiles else f"q_num={self.q_num}"
        return 'f_num={} | {} | embed_dim={} | mode={}'.format(self.f_num, q_repr, self.embedding_dim, self.mode)
