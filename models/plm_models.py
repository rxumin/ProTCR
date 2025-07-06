import numpy as np
from torch.nn import CrossEntropyLoss
import copy
from transformers import T5PreTrainedModel, T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map
from typing import Union, Optional, Tuple
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel

class Biaffine(nn.Module):
    def __init__(self, in_size, out_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.out_size = out_size
        self.U = torch.nn.Parameter(torch.randn(in_size + int(bias_x), out_size, in_size + int(bias_y)))
        # self.U1 = self.U.view(size=(in_size + int(bias_x),-1))
        # U.shape = [in_size,out_size,in_size]

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)

        print(x.shape, y.shape)
        bilinar_mapping = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)
        return bilinar_mapping


class GatedBiaffine(nn.Module):
    def __init__(self, in_size, out_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.out_size = out_size
        self.U = nn.Parameter(torch.randn(in_size + int(bias_x), out_size, in_size + int(bias_y)))
        self.W_gate = nn.Parameter(torch.randn(in_size + int(bias_x), in_size + int(bias_y)))
        self.b_gate = nn.Parameter(torch.randn(1))

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)

        # Compute the biaffine transformation
        bilinear_mapping = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)

        # Compute the gating mechanism
        gating_scores = torch.einsum('bxi,ij,byj->bxy', x, self.W_gate, y)
        gating_scores = torch.sigmoid(gating_scores + self.b_gate)

        # Apply the gate to the biaffine output
        gated_output = bilinear_mapping * gating_scores.unsqueeze(-1)

        return gated_output


class SEBlock(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.input_channels = input_channels
        reduced_channels = input_channels // reduction_ratio

        # 全连接层用于降维
        self.fc_reduce = nn.Linear(input_channels, reduced_channels)
        # 全连接层用于升维
        self.fc_expand = nn.Linear(reduced_channels, input_channels)

    def forward(self, x):
        # Squeeze: 全局平均池化，将每个通道的空间维度压缩为1
        batch_size, channels, _, _ = x.size()
        squeeze = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, channels)

        # Excitation: 全连接层降维后接ReLU，然后升维后接Sigmoid
        excitation = F.relu(self.fc_reduce(squeeze))
        excitation = torch.sigmoid(self.fc_expand(excitation))

        # 重塑为原始输入的形状，以便进行通道加权
        excitation = excitation.view(batch_size, channels, 1, 1)

        # Scale: 通过乘以输入的特征图来缩放
        return x * excitation


class SEBlock_conv11(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock_conv11, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                              bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                            bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class MultiKernelCNNWithSE(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_sizes=[2, 3, 4, 5], reduction_ratio=16):
        super(MultiKernelCNNWithSE, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=k, padding=k // 2),
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                SEBlock(output_channels, reduction_ratio=reduction_ratio),
                nn.AdaptiveMaxPool2d((1, 1))
            ) for k in kernel_sizes
        ])

        # Assuming square input and pooling reduces by factor of 2
        self.fc = nn.Linear(output_channels * len(kernel_sizes), output_channels * len(kernel_sizes))

    def forward(self, x):
        # Apply each convolution + SE block + pooling in parallel
        outputs = [conv(x) for conv in self.convs]
        # Flatten and concatenate outputs along the channel dimension
        outputs = torch.cat([o.view(o.size(0), -1) for o in outputs], dim=1)
        # Fully connected layer to classify
        outputs = self.fc(outputs)
        return outputs



class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


class ConvolutionLayer(nn.Module):
    def __init__(self, input_size, channels, dilation, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(),
        )

        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)

        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = F.gelu(x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            z = torch.ones_like(x[..., :1])
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1) # 2 11 513
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1) # 2 28 513
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y) # 2 512 11 28
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1) # 2 11 28 512

        return s


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class CoPredictor(nn.Module):
    def __init__(self, hid_size, biaffine_size, channels, ffnn_hid_size, dropout=0):
        super().__init__()
        self.mlp1 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.mlp2 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=biaffine_size, bias_x=True, bias_y=True)
        self.mlp_rel = MLP(channels, ffnn_hid_size, dropout=dropout)
        self.linear = nn.Linear(ffnn_hid_size, biaffine_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, z):
        h = self.dropout(self.mlp1(x))
        t = self.dropout(self.mlp2(y))
        o1 = self.biaffine(h, t)
        return o1

class AttentionBasedFeatureFusion(nn.Module):
    def __init__(self, layer_count, hidden_dim=128):
        super(AttentionBasedFeatureFusion, self).__init__()
        self.layer_count = layer_count
        self.hidden_dim = hidden_dim
        self.query = nn.LazyLinear(hidden_dim)
        self.key = nn.LazyLinear(hidden_dim)
        self.value = nn.LazyLinear(hidden_dim)
        self.weight_network = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.Sigmoid(),
            nn.LazyLinear(layer_count)
        )

    def forward(self, layer_outputs):
        # 检查输入的层数是否与初始化时期望的一致
        assert len(layer_outputs) == self.layer_count, "输入层的数量与期望的层数不匹配。"
        # 取出最后的维度 1024
        self.feature_dim = layer_outputs[0].shape[-1]
        # for x in layer_outputs:
        #     print(x.max())
        # 使用query、key和value转换每一层的平均输出 2*12*1024
        means=[]
        for x in layer_outputs:
            means_value=torch.mean(x,dim=1)
            means.append(means_value)

        avg_outputs = torch.stack([torch.mean(x, dim=1) for x in layer_outputs],
                                  dim=1)  # (batch_size, layer_count, feature_dim)
        # avg_outputs = torch.stack([output[:,0] for output in layer_outputs], dim=1)
        queries = self.query(avg_outputs)  # (batch_size, layer_count, hidden_dim)
        keys = self.key(avg_outputs)  # (batch_size, layer_count, hidden_dim)
        values = self.value(avg_outputs)  # (batch_size, layer_count, hidden_dim)

        # 计算注意力分数
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.feature_dim ** 0.5)  # Scale
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, layer_count, layer_count)

        # 计算加权的值
        weighted_values = torch.matmul(attention_weights, values)  # (batch_size, layer_count, hidden_dim)

        # 使用权重网络计算每层的最终权重
        # z = weighted_values.reshape(-1, self.layer_count * self.hidden_dim)  # 2*12*1024
        final_weights = self.weight_network(weighted_values.reshape(-1, self.layer_count * self.hidden_dim))
        z = F.softmax(final_weights, dim=-1)
        final_weights = F.softmax(final_weights, dim=-1).reshape(-1, self.layer_count, 1,
                                                                 1)  # (batch_size, layer_count, 1, 1)

        # 将层输出堆叠成一个新的维度，形状变为(batch_size, layer_count, seq_len, feature_dim)
        stacked_outputs = torch.stack(layer_outputs, dim=1)

        # 使用广播机制进行加权融合
        fused_feature = torch.sum(final_weights * stacked_outputs, dim=1)

        return fused_feature


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class FocalLossBinary(nn.Module):
    def __init__(self, alpha=1., gamma=2., redution='mean'):
        super(FocalLossBinary, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.redution = redution

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            self.alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = self.alpha_t * loss

        # Check self.redution option and return loss accordingly
        if self.redution == "none":
            pass
        elif self.redution == "mean":
            loss = loss.mean()
        elif self.redution == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'self.redution': '{self.redution} \n Supported self.redution modes: 'none', 'mean', 'sum'"
            )
        return loss

class Prott5(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"encoder.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        # 共享的嵌入层
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.num_labels = config.num_labels
        encoder_config = copy.deepcopy(config)
        # 不使用缓存
        encoder_config.use_cache = False
        # 只使用编码器
        encoder_config.is_encoder_decoder = False
        # 创建编码器
        self.encoder = T5Stack(encoder_config, self.shared)
        # 对特征进行标准化
        self.layer_norm = T5LayerNorm(hidden_size=config.d_model)
        # 随机丢掉神经元，减少过拟合
        self.dropout = nn.Dropout(config.dropout_rate)
        # 注意力
        self.adaptive_fusion = AttentionBasedFeatureFusion(layer_count=12, hidden_dim=1024)
        # GRU
        self.gru = nn.GRU(config.d_model, config.d_model // 2, bidirectional=True, num_layers=1)
        self.classifier1 = nn.Linear(config.d_model, config.num_labels)

        conv_input_size = 1024
        conv_hid_size = 512
        dilation = [1, 2, 3]
        conv_dropout = emb_dropout = 0.2
        biaffine_size = 512
        ffnn_hid_size = 256
        out_dropout = 0.1
        hidden_size = 1024
        self.convLayer = ConvolutionLayer(conv_input_size, conv_hid_size, dilation, conv_dropout)
        self.dropout = nn.Dropout(emb_dropout)
        self.predictor = CoPredictor(hidden_size, biaffine_size,
                                     conv_hid_size * len(dilation), ffnn_hid_size,
                                     out_dropout)
        self.global_pool = nn.Conv2d(in_channels=biaffine_size, out_channels=1, kernel_size=1)

        self.classifier2 = nn.Linear(biaffine_size, config.num_labels)
        self.bn = nn.LazyBatchNorm1d()

        self.classifier = nn.LazyLinear(config.num_labels)

        # 将2种不同的token映射过去
        self.token_type_embedding = nn.Embedding(2, config.d_model)
        # 投影到低纬度，然后引入非线性特征，使得模型学习复杂的特征，再次投影回原始的维度
        self.token_type_linear = nn.Sequential(nn.Linear(config.d_model, config.d_model//2),
                                               nn.ReLU(),
                                               nn.Linear(config.d_model//2, config.d_model))

        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        token_type_embedding = self.token_type_embedding(token_type_ids) # 2*39 -> 2*39*1024
        token_type_embedding = self.token_type_linear(token_type_embedding)

        # 提取最后 12 个隐藏状态 2*39*1024
        layer_hidden_states = list(encoder_outputs.hidden_states[-12:])
        for i in range(len(layer_hidden_states[:-1])):     # len()=12

            # 将隐藏状态的特征于token_type_embedding融合
            layer_hidden_states[i] = layer_hidden_states[i] + token_type_embedding
            # 随机丢弃神经元
            layer_hidden_states[i] = self.dropout(self.layer_norm(layer_hidden_states[i]))
        hidden_states = self.adaptive_fusion(layer_hidden_states)

        # 分支一
        x1, _ = self.gru(hidden_states)
        x1, _ = torch.max(x1, dim=1)

        # 分支二
        outputs = self.predictor(hidden_states[:,:11,:], hidden_states[:,11:,:], None)
        x2 = self.global_pool(outputs.permute(0, 3, 1, 2))
        x2 = x2.squeeze(dim=-1).view(x2.shape[0], -1)
        x2 = self.bn(x2)
        # x2 = self.classifier2(x2.squeeze())

        # logits = x1 + x2
        i = torch.cat([x1, x2],dim=1)
        logits = self.classifier(torch.cat([x1,x2], dim=1))
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            b = logits.view(-1, self.num_labels)
            a = labels.view(-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return ModelOutput(
            loss=loss,
            logits=logits
        )

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)
