# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Bert for pretraining."""
import numpy as np

import mindspore.nn as nn
from mindspore.common.initializer import initializer, TruncatedNormal
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import Parameter, ParameterTuple
from mindspore import dtype as mstype
from mindspore.nn import DistributedGradReducer
from mindspore.context import ParallelMode
from mindspore.communication import get_group_size
from mindspore import context
from .bert_model import BertModel

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0

clip_grad = ops.MultitypeFuncGraph("clip_grad")


# pylint: disable=consider-using-in
@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type != 0 and clip_type != 1:
        return grad
    dt = ops.dtype(grad)
    if clip_type == 0:
        new_grad = ops.clip_by_value(grad, ops.cast(ops.tuple_to_array((-clip_value,)), dt),
                                     ops.cast(ops.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, ops.cast(ops.tuple_to_array((clip_value,)), dt))
    return new_grad


class GetMaskedLMOutput(nn.Cell):
    """
    Get masked lm output.

    Args:
        config (BertConfig): The config of BertModel.

    Returns:
        Tensor, masked lm output.
    """
    def __init__(self, config):
        super(GetMaskedLMOutput, self).__init__()
        self.width = config.hidden_size
        self.reshape = ops.Reshape()
        self.gather = ops.GatherV2()

        weight_init = TruncatedNormal(config.initializer_range)
        self.dense = nn.Dense(self.width,
                              config.hidden_size,
                              weight_init=weight_init,
                              activation=config.hidden_act).to_float(config.compute_type)
        self.layernorm = nn.LayerNorm((config.hidden_size,)).to_float(config.compute_type)
        self.output_bias = Parameter(
            initializer(
                'zero',
                config.vocab_size),
            name='output_bias')
        self.matmul = ops.MatMul(transpose_b=True)
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.shape_flat_offsets = (-1, 1)
        self.rng = Tensor(np.array(range(0, config.batch_size)).astype(np.int32))
        self.last_idx = (-1,)
        self.shape_flat_sequence_tensor = (config.batch_size * config.seq_length, self.width)
        self.seq_length_tensor = Tensor(np.array((config.seq_length,)).astype(np.int32))
        self.cast = ops.Cast()
        self.compute_type = config.compute_type
        self.dtype = config.dtype

    def construct(self,
                  input_tensor,
                  output_weights,
                  positions):
        """construct masked lm output"""
        flat_offsets = self.reshape(
            self.rng * self.seq_length_tensor, self.shape_flat_offsets)
        flat_position = self.reshape(positions + flat_offsets, self.last_idx)
        flat_sequence_tensor = self.reshape(input_tensor, self.shape_flat_sequence_tensor)
        input_tensor = self.gather(flat_sequence_tensor, flat_position, 0)
        input_tensor = self.cast(input_tensor, self.compute_type)
        output_weights = self.cast(output_weights, self.compute_type)
        input_tensor = self.dense(input_tensor)
        input_tensor = self.layernorm(input_tensor)
        logits = self.matmul(input_tensor, output_weights)
        logits = self.cast(logits, self.dtype)
        logits = logits + self.output_bias
        log_probs = self.log_softmax(logits)
        return log_probs


class GetNextSentenceOutput(nn.Cell):
    """
    Get next sentence output.

    Args:
        config (BertConfig): The config of Bert.

    Returns:
        Tensor, next sentence output.
    """
    def __init__(self, config):
        super(GetNextSentenceOutput, self).__init__()
        self.log_softmax = ops.LogSoftmax()
        weight_init = TruncatedNormal(config.initializer_range)
        self.dense = nn.Dense(config.hidden_size, 2,
                              weight_init=weight_init, has_bias=True).to_float(config.compute_type)
        self.dtype = config.dtype
        self.cast = ops.Cast()

    def construct(self, input_tensor):
        logits = self.dense(input_tensor)
        logits = self.cast(logits, self.dtype)
        log_prob = self.log_softmax(logits)
        return log_prob


class BertPreTraining(nn.Cell):
    """
    Bert pretraining network.

    Args:
        config (BertConfig): The config of BertModel.
        is_training (bool): Specifies whether to use the training mode.
        use_one_hot_embeddings (bool): Specifies whether to use one-hot for embeddings.

    Returns:
        Tensor, prediction_scores, seq_relationship_score.
    """
    def __init__(self, config, is_training, use_one_hot_embeddings):
        super(BertPreTraining, self).__init__()
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cls1 = GetMaskedLMOutput(config)
        self.cls2 = GetNextSentenceOutput(config)

    def construct(self, input_ids, input_mask, token_type_id,
                  masked_lm_positions):
        sequence_output, pooled_output, embedding_table = \
            self.bert(input_ids, token_type_id, input_mask)
        prediction_scores = self.cls1(sequence_output,
                                      embedding_table,
                                      masked_lm_positions)
        seq_relationship_score = self.cls2(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPretrainingLoss(nn.Cell):
    """
    Provide bert pre-training loss.

    Args:
        config (BertConfig): The config of BertModel.

    Returns:
        Tensor, total loss.
    """
    def __init__(self, config):
        super(BertPretrainingLoss, self).__init__()
        self.vocab_size = config.vocab_size
        self.onehot = ops.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.reduce_sum = ops.ReduceSum()
        self.reduce_mean = ops.ReduceMean()
        self.reshape = ops.Reshape()
        self.last_idx = (-1,)
        self.neg = ops.Neg()
        self.cast = ops.Cast()

    def construct(self, prediction_scores, seq_relationship_score, masked_lm_ids,
                  masked_lm_weights, next_sentence_labels):
        """Defines the computation performed."""
        label_ids = self.reshape(masked_lm_ids, self.last_idx)
        label_weights = self.cast(self.reshape(masked_lm_weights, self.last_idx), mstype.float32)
        one_hot_labels = self.onehot(label_ids, self.vocab_size, self.on_value, self.off_value)

        per_example_loss = self.neg(self.reduce_sum(prediction_scores * one_hot_labels, self.last_idx))
        numerator = self.reduce_sum(label_weights * per_example_loss, ())
        denominator = self.reduce_sum(label_weights, ()) + self.cast(ops.tuple_to_array((1e-5,)), mstype.float32)
        masked_lm_loss = numerator / denominator

        # next_sentence_loss
        labels = self.reshape(next_sentence_labels, self.last_idx)
        one_hot_labels = self.onehot(labels, 2, self.on_value, self.off_value)
        per_example_loss = self.neg(self.reduce_sum(
            one_hot_labels * seq_relationship_score, self.last_idx))
        next_sentence_loss = self.reduce_mean(per_example_loss, self.last_idx)

        # total_loss
        total_loss = masked_lm_loss + next_sentence_loss

        return total_loss


class BertNetworkWithLoss(nn.Cell):
    """
    Provide bert pre-training loss through network.

    Args:
        config (BertConfig): The config of BertModel.
        is_training (bool): Specifies whether to use the training mode.
        use_one_hot_embeddings (bool): Specifies whether to use one-hot for embeddings. Default: False.

    Returns:
        Tensor, the loss of the network.
    """
    def __init__(self, config, is_training, use_one_hot_embeddings=False):
        super(BertNetworkWithLoss, self).__init__()
        self.bert = BertPreTraining(config, is_training, use_one_hot_embeddings)
        self.loss = BertPretrainingLoss(config)
        self.cast = ops.Cast()

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  next_sentence_labels,
                  masked_lm_positions,
                  masked_lm_ids,
                  masked_lm_weights):
        """construct bertworkwitlosscell"""
        prediction_scores, seq_relationship_score = \
            self.bert(input_ids, input_mask, token_type_id, masked_lm_positions)
        total_loss = self.loss(prediction_scores, seq_relationship_score,
                               masked_lm_ids, masked_lm_weights, next_sentence_labels)
        return self.cast(total_loss, mstype.float32)


class BertTrainOneStepCell(nn.Cell):
    """
    Encapsulation class of bert network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """
    def __init__(self, network, optimizer, sens=1.0):
        super(BertTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = None
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("mirror_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)

        self.cast = ops.Cast()
        self.hyper_map = ops.HyperMap()

    def set_sens(self, value):
        self.sens = value

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  next_sentence_labels,
                  masked_lm_positions,
                  masked_lm_ids,
                  masked_lm_weights):
        """Defines the computation performed."""
        weights = self.weights

        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            next_sentence_labels,
                            masked_lm_positions,
                            masked_lm_ids,
                            masked_lm_weights)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 next_sentence_labels,
                                                 masked_lm_positions,
                                                 masked_lm_ids,
                                                 masked_lm_weights,
                                                 self.cast(ops.tuple_to_array((self.sens,)),
                                                           mstype.float32))
        grads = self.hyper_map(ops.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        succ = self.optimizer(grads)
        return ops.depend(loss, succ)


grad_scale = ops.MultitypeFuncGraph("grad_scale")
reciprocal = ops.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


class BertTrainOneStepWithLossScaleCell(nn.Cell):
    """
    Encapsulation class of bert network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """
    def __init__(self, network, optimizer, scale_update_cell=None):
        super(BertTrainOneStepWithLossScaleCell, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = ops.GradOperation(
            get_by_list=True,
            sens_param=True)
        self.reducer_flag = False
        self.allreduce = ops.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = ops.identity
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = ops.Cast()
        self.alloc_status = ops.NPUAllocFloatStatus()
        self.get_status = ops.NPUGetFloatStatus()
        self.clear_before_grad = ops.NPUClearFloatStatus()
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        self.depend_parameter_use = ops.ControlDepend(depend_mode=1)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = ops.LessEqual()
        self.hyper_map = ops.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32),
                                        name="loss_scale")

    @ops.add_flags(has_effect=True)
    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  next_sentence_labels,
                  masked_lm_positions,
                  masked_lm_ids,
                  masked_lm_weights,
                  sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            next_sentence_labels,
                            masked_lm_positions,
                            masked_lm_ids,
                            masked_lm_weights)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        # alloc status and clear should be right before gradoperation
        init = self.alloc_status()
        self.clear_before_grad(init)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 next_sentence_labels,
                                                 masked_lm_positions,
                                                 masked_lm_ids,
                                                 masked_lm_weights,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(ops.partial(grad_scale, scaling_sens * self.degree), grads)
        grads = self.hyper_map(ops.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        self.get_status(init)
        flag_sum = self.reduce_sum(init, (0,))
        if self.is_distributed:
            # sum overflow flag over devices
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        ret = (loss, cond, scaling_sens)
        return ops.depend(ret, succ)
