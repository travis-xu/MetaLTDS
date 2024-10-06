from transformers import GPT2Model, GPT2Tokenizer, GPT2PreTrainedModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Union, List
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
from fnmatch import fnmatch
from utils.utils_CL import calculate_task_adapter_id, calculate_adapter_num
from utils.config import *


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a ``__getitem__`` that allows indexing by integer or slice (like
    a tuple) or strings (like a dictionary) that will ignore the ``None`` attributes. Otherwise behaves like a
    regular python dictionary.
    .. warning::
        You can't unpack a :obj:`ModelOutput` directly. Use the :meth:`~transformers.file_utils.ModelOutput.to_tuple`
        method to convert it to a tuple before.
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        assert len(class_fields), f"{self.__class__.__name__} has no fields."
        assert all(
            field.default is None for field in class_fields[1:]
        ), f"{self.__class__.__name__} should not have more than one required field."

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            try:
                iterator = iter(first_field)
                first_field_iterator = True
            except TypeError:
                first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (
                            not isinstance(element, (list, tuple))
                            or not len(element) == 2
                            or not isinstance(element[0], str)
                    ):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).
            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            If :obj:`past_key_values` is used only the last hidden-state of the sequences of shape
            :obj:`(batch_size, 1, hidden_size)` is output.
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).
            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Adapter(nn.Module):
    def __init__(self, config, bottleneck=0):
        super(Adapter, self).__init__()
        nx = config.n_embd
        self.ln = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        # self.project_down = nn.Linear(nx, bottleneck)
        self.fc1 = nn.Linear(nx, bottleneck)
        self.relu = nn.ReLU()
        # self.project_up = nn.Linear(bottleneck, nx)
        self.fc2 = nn.Linear(bottleneck, nx)

    def forward(self, x):
        x_ = self.ln(x)
        # x_ = self.project_down(x_)
        x_ = self.fc1(x_)
        x_ = self.relu(x_)
        # x_ = self.project_up(x_)
        x_ = self.fc2(x_)
        x = x + x_  # residual connection
        return x

# CLASSIC
# class GPT2Adapter(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.fc1=torch.nn.Linear(config.bert_hidden_size,config.bert_adapter_size)
#         self.fc2=torch.nn.Linear(config.bert_adapter_size,config.bert_hidden_size)
#         if  config.use_gelu: self.activation = torch.nn.GELU()
#         else: self.activation = torch.nn.ReLU()
#         print('GPT2Adapter')
#
#     def forward(self,x):
#
#         h=self.activation(self.fc1(x))
#         h=self.activation(self.fc2(h))
#
#         return x + h
#         # return h
#
#     def squash(self, input_tensor, dim=-1,epsilon=1e-16): # 0 will happen in our case, has to add an epsilon
#         squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
#         squared_norm = squared_norm + epsilon
#         scale = squared_norm / (1 + squared_norm)
#         return scale * input_tensor / torch.sqrt(squared_norm)

# CLASSIC GPT2AdapterMask
class AdapterMask(Adapter):
    def __init__(self, config, args, adapter_id=None):
        super().__init__(config, args.bottleneck_size_list[adapter_id] if args.split_mask else args.bottleneck_size)
        if args.split_mask:
            self.efc1 = torch.nn.Embedding(args.task_adapter_list[adapter_id], args.bottleneck_size_list[adapter_id])
            self.efc2 = torch.nn.Embedding(args.task_adapter_list[adapter_id], config.n_embd)
        else:
            self.efc1=torch.nn.Embedding(args.number_of_adpt,args.bottleneck_size)
            self.efc2=torch.nn.Embedding(args.number_of_adpt,config.n_embd)

        self.gate=torch.nn.Sigmoid()
        self.config = config
        # print('GPT2AdapterMask')

    def forward(self, x, t, s=None, mask_layer_pre=None):
        if mask_layer_pre is not None and s is not None:
            gfc1_pre, gfc2_pre = mask_layer_pre
            gfc1_cur, gfc2_cur = self.mask(t=t, s=s)
            gfc1 = torch.max(gfc1_pre, gfc1_cur)
            gfc2 = torch.max(gfc2_pre, gfc2_cur)
            # for key, value in model.mask_pre.items():
            #     model.mask_pre[key] = torch.max(model.mask_pre[key], mask[key])
        elif mask_layer_pre is not None:
            assert s is None
            gfc1, gfc2 = mask_layer_pre
        else:
            assert mask_layer_pre is None
            gfc1,gfc2=self.mask(t=t,s=s)
        h = self.get_feature(gfc1,gfc2,x)
        return x + h

    def get_feature(self,gfc1,gfc2,x):
        # x_ = self.ln(x)
        # x_ = self.project_down(x_)
        # x_ = self.relu(x_)
        # h = h * gfc1.expand_as(h)
        # x_ = self.project_up(x_)
        # h = h * gfc2.expand_as(h)
        h=self.relu(self.fc1(self.ln(x)))
        h=h*gfc1.expand_as(h)

        h=self.fc2(h)
        h=h*gfc2.expand_as(h)
        return h

    def mask(self,t,s=1):
        efc1 = self.efc1(torch.LongTensor([t]).cuda())
        efc2 = self.efc2(torch.LongTensor([t]).cuda())
        gfc1 = self.gate(s * efc1)
        gfc2 = self.gate(s * efc2)

        gfc1 = gfc1.view(-1, 1, gfc1.size(-1))
        gfc2 = gfc2.view(-1, 1, gfc2.size(-1))
        return [gfc1, gfc2]

class MixAdapterMask(nn.Module):
    def __init__(self, config, args, adapter_num=-1):
        super(MixAdapterMask, self).__init__()
        # 20 adapters with task_id 0--19, when task_id==-1 means dont use adapter
        self.mixadaptermask = nn.ModuleList([AdapterMask(config, args, adapter_id) for adapter_id in range(adapter_num)])

    def forward(self, x, t=-1, s=None, mask_layer_pre=None):
        if t == -1:
            return x
        else:
            if type(t)==list:
                adapter_id, _ = calculate_task_adapter_id(t[0])
                adapter_task_id = [calculate_task_adapter_id(t_id)[-1] for t_id in t]
            else:
                adapter_id, adapter_task_id = calculate_task_adapter_id(t)
            return self.mixadaptermask[adapter_id](x, adapter_task_id, s, mask_layer_pre)

    def mask(self,t,s=1):
        if type(t) == list:
            adapter_id, _ = calculate_task_adapter_id(t[0])
            adapter_task_id = [calculate_task_adapter_id(t_id)[-1] for t_id in t]
        else:
            adapter_id, adapter_task_id = calculate_task_adapter_id(t)
        # adapter_id, adapter_task_id = calculate_task_adapter_id(t)
        return self.mixadaptermask[adapter_id].mask(adapter_task_id, s)


class GPT2Adapter(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()
        self.config = config

    def get_output_embeddings(self):
        return self.lm_head

    def add_adapters(self, args):
        # bottleneck_size = args.bottleneck_size
        if args.mask:
            if args.split_mask:
                adapter_num = args.num_of_adapter
                self.adapter_blocks = nn.ModuleList(
                    [MixAdapterMask(self.config, args, adapter_num) for _ in range(self.config.n_layer)])
            else:
                self.adapter_blocks = nn.ModuleList([AdapterMask(self.config, args) for _ in range(self.config.n_layer)])
        else:
            self.adapter_blocks = nn.ModuleList(
                [Adapter(self.config, args.bottleneck_size) for _ in range(self.config.n_layer)])
        # self.adapter_blocks = nn.ModuleList(
        #     [MixAdapter(self.config, bottleneck_size, adapter_num) for _ in range(self.config.n_layer)])
        # self.init_weights()

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create postion_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task_id=-1,
            s=None,
            masks_pre=None,
            **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        """
        if "past" in kwargs:
            warnings.warn(
                "The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("past")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # transformer_outputs = self.transformer(
        #     input_ids,
        #     past_key_values=past_key_values,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids,
        #     position_ids=position_ids,
        #     head_mask=head_mask,
        #     inputs_embeds=inputs_embeds,
        #     encoder_hidden_states=encoder_hidden_states,
        #     encoder_attention_mask=encoder_attention_mask,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        ### OVERLOADING THIS FUNCTION
        if "past" in kwargs:
            warnings.warn(
                "The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("past")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        output_attentions = output_attentions if output_attentions is not None else self.transformer.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.transformer.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.transformer.config.use_cache
        return_dict = return_dict if return_dict is not None else self.transformer.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.transformer.h)
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.transformer.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.transformer.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.transformer.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.transformer.get_head_mask(head_mask, self.transformer.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.transformer.wte(input_ids)
        position_embeds = self.transformer.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.transformer.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.transformer.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        if (complex):
            img_part = torch.zeros_like(hidden_states)
        for i, (block, layer_past, adapter) in enumerate(zip(self.transformer.h, past_key_values, self.adapter_blocks)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            if getattr(self.transformer.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # checkpointing only works with tuple returns, not with lists
                        return tuple(output for output in module(*inputs, use_cache, output_attentions))

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    layer_past,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # outputs[0] = adapter(outputs[0], task_id=task_id)
            # s, t = None, None
            # if 't' in kwargs: t = kwargs['t']
            # if 's' in kwargs:
            #     s = kwargs['s']

            if masks_pre is not None and s is not None:  # mask_CIL
                fc1_key = 'adapter_blocks.' + str(i) + '.fc1'  # gfc1
                fc2_key = 'adapter_blocks.' + str(i) + '.fc2'  # gfc2
                mask_layer_pre = (masks_pre[fc1_key], masks_pre[fc2_key])
                outputs[0] = adapter(outputs[0], task_id, s, mask_layer_pre=mask_layer_pre)
            elif masks_pre is not None: # HAT mask: retrain
                fc1_key = 'adapter_blocks.' + str(i) + '.fc1'  # gfc1
                fc2_key = 'adapter_blocks.' + str(i) + '.fc2'  # gfc2
                mask_layer_pre = (masks_pre[fc1_key], masks_pre[fc2_key])
                outputs[0] = adapter(outputs[0], task_id, mask_layer_pre=mask_layer_pre)
            elif s is not None:   # HAT mask: train/eval
                outputs[0] = adapter(outputs[0], task_id, s)
            else:
                outputs[0] = adapter(outputs[0])    ##
            # else:
            #     outputs[0] = adapter(outputs[0])    ##

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_attentions = all_attentions + (outputs[2],)

        hidden_states = self.transformer.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            transformer_outputs = tuple(
                v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)
        else:
            transformer_outputs = BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=presents,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
            )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


    def mask(self, t, s, appendix=''):
        if hparams.split_mask:
            if appendix == '':
                adapter_id, adapter_task_id = calculate_task_adapter_id(t)
                appendix = f'.mixadaptermask.{adapter_id}'
        else:
            appendix = ''
        masks = {}
        for layer_id in range(self.config.n_layer):
            # fc1_key = 'bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_mask.fc1' #gfc1
            fc1_key = 'adapter_blocks.' + str(layer_id) + appendix + '.fc1'  # gfc1
            fc2_key = 'adapter_blocks.' + str(layer_id) + appendix + '.fc2'  # gfc2
            masks[fc1_key], masks[fc2_key] = self.adapter_blocks[layer_id].mask(t, s)
            # fc1_key = 'bert.encoder.layer.'+str(layer_id)+'.output.adapter_mask.fc1' #gfc1
            # masks[fc1_key],masks[fc2_key] = self.bert.encoder.layer[layer_id].output.adapter_mask.mask(t,s)
        return masks


    def get_view_for(self,n,p,masks,appendix=''):
        for layer_id in range(self.config.n_layer):
            if n=='adapter_blocks.'+str(layer_id)+appendix+'.fc1.weight':
                return masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
            elif n=='adapter_blocks.'+str(layer_id)+appendix+'.fc1.bias':
                return masks[n.replace('.bias','')].data.view(-1)
            elif n=='adapter_blocks.'+str(layer_id)+appendix+'.fc2.weight':
                post=masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
                pre=masks[n.replace('.weight','').replace('fc2','fc1')].data.view(1,-1).expand_as(p)
                return torch.min(post,pre)
            elif n=='adapter_blocks.'+str(layer_id)+appendix+'.fc2.bias':
                return masks[n.replace('.bias','')].data.view(-1)
        return None

    def reset_mask(self, hparams, t):
        for n,p in self.adapter_blocks.named_parameters():
            if hparams.todcl_mask:
                if '.efc2' in n:
                    p.data.fill_(np.inf)
                elif '.efc1' in n:
                    p.data.fill_(-np.inf)   # np.inf
                    for i in range(hparams.number_of_adpt):
                        p.data[i, (i * 50):(i * 50 + 50)].fill_(np.inf)
                        # p.data[i, (i * 50):(i * 50 + 50)].copy_(torch.full((50,), np.inf))    # 6
            elif hparams.expand_mask:
                if '.efc2' in n:
                    p.data.fill_(np.inf)
                # self.efc2.weight.data.copy_(torch.full((hparams.number_of_adpt, self.config.n_embd), np.inf))
                elif '.efc1' in n:
                    p.data[t, hparams.cur_bottleneck_size:].fill_(-np.inf)
                    # for i in range(hparams.number_of_adpt):
                    #     p.data[i, (i * 50 + 50):].fill_(-np.inf)
                        # self.efc1.weight.data[i, (i * 50 + 50):].copy_(torch.full_like(self.efc1.weight.data[i, (i * 50 + 50):], -np.inf))
            elif hparams.split_mask:
                if '.efc2' in n:
                    p.data.fill_(np.inf)


