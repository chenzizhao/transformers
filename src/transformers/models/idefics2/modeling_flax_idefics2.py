from typing import Dict, List, Optional, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

from ...modeling_flax_outputs import FlaxBaseModelOutput
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel
from ...modeling_outputs import ModelOutput
from .configuration_idefics2 import Idefics2Config, Idefics2VisionConfig


@flax.struct.dataclass
class FlaxIdefics2BaseModelOutputWithPast(ModelOutput):
    last_hidden_state: jnp.ndarray = None
    past_key_values: Optional[Dict[str, jnp.ndarray]] = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None
    image_hidden_states: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxIdefics2CausalLMOutputWithPast(ModelOutput):
    loss: Optional[jnp.ndarray] = None
    logits: jnp.ndarray = None
    past_key_values: Optional[List[jnp.ndarray]] = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None
    image_hidden_states: Optional[Tuple[jnp.ndarray]] = None


# copied from FlaxCLIPVisionEmbeddings
class FlaxIdefics2VisionEmbeddings(nn.Module):
    config: Idefics2VisionConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embed_dim = self.config.hidden_size
        self.image_size = self.config.image_size
        self.patch_size = self.config.patch_size

        self.patch_embedding = nn.Conv(
            self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            kernel_init=nn.initializers.normal(),
        )

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embed(self.num_positions, self.embed_dim, embedding_init=nn.initializers.normal())

    def __call__(self, pixel_values: jnp.ndarray, patch_attention_mask: jnp.ndarray):
        batch_size, _, max_im_h, max_im_w = pixel_values.shape

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = jnp.transpose(patch_embeds, (0, 2, 3, 1))
        embeddings = jnp.reshape(embeddings, (batch_size, -1, self.embed_dim))

        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
        boundaries = jnp.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
        position_ids = jnp.zeros((batch_size, max_nb_patches_h * max_nb_patches_w), dtype=jnp.int32)

        def compute_position_ids(p_attn_mask):
            nb_patches_h = jnp.sum(p_attn_mask[:, 0])
            nb_patches_w = jnp.sum(p_attn_mask[0])

            fractional_coords_h = jnp.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = jnp.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = jnp.digitize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = jnp.digitize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            return pos_ids[jnp.ravel(p_attn_mask)]

        position_ids = jax.vmap(compute_position_ids)(patch_attention_mask)
        position_ids = jax.device_put(position_ids, pixel_values.device())
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


# copied from FlaxCLIPVisionAttention
class FlaxIdefics2VisionAttention(nn.Module):
    config: Idefics2VisionConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=nn.initializers.normal(0.01))
        self.v_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=nn.initializers.normal(0.01))
        self.q_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=nn.initializers.normal(0.01))
        self.out_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=nn.initializers.normal(0.01))

        # Ignore copy
        self.is_causal = False

    def _split_heads(self, hidden_states: jnp.ndarray):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states: jnp.ndarray):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[Tuple[jnp.ndarray]]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # split heads
        # query_states = query_states.reshape(batch_size, q_len, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        # key_states = key_states.reshape(batch_size, q_len, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        # value_states = value_states.reshape(batch_size, q_len, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)

        if attention_mask is not None:
            # TODO(czz):?
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
        else:
            attention_bias = None

        dropout_rng = None
        if self.dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            query=query_states,
            key=key_states,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout,
            deterministic=False,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


# Copied from FlaxCLIPMLP
class FlaxIdefics2VisionMLP(nn.Module):
    config: Idefics2VisionConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        config = self.config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Dense(
            config.hidden_size, config.intermediate_size, dtype=self.dtype, kernel_init=nn.initializers.normal(0.01)
        )
        self.fc2 = nn.Dense(
            config.intermediate_size, config.hidden_size, dtype=self.dtype, kernel_init=nn.initializers.normal(0.01)
        )

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class FlaxIdefics2MLP(nn.Module):
    dtype: jnp.dtype = jnp.float32
    hidden_size: int
    intermediate_size: int
    out_size: int
    hidden_act: str

    def setup(self):
        self.gate_proj = nn.Dense(
            self.hidden_size, self.intermediate_size, bias=False, kernel_init=nn.initializers.normal(0.01)
        )
        self.up_proj = nn.Dense(
            self.hidden_size, self.intermediate_size, bias=False, kernel_init=nn.initializers.normal(0.01)
        )
        self.down_proj = nn.Dense(
            self.intermediate_size, self.output_size, bias=False, kernel_init=nn.initializers.normal(0.01)
        )
        self.act_fn = ACT2FN[self.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class FlaxIdefics2MultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    config: Idefics2VisionConfig

    def setup(self):
        # use in set up intes
        config = self.config
        self.probe = self.param("probe", nn.initializers.normal(), (1, 1, config.hidden_size))
        self.attention = nn.MultiHeadDotProductAttention(
            config.hidden_size, config.num_attention_heads, batch_first=True
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Ignore copy
        self.mlp = FlaxIdefics2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            output_size=config.hidden_size,
        )

    def __call__(self, hidden_state: jnp.ndarray) -> jnp.ndarray:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class FlaxIdefics2EncoderLayer(nn.Module):
    # TODO: double check if this is vision config or idefics2 config
    config: Idefics2VisionConfig

    def setup(self):
        config = self.config

        self.embed_dim = config.hidden_size
        self.self_attn = FlaxIdefics2VisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = FlaxIdefics2VisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Copied from transformers.models.siglip.modeling_siglip.SiglipEncoderLayer.forward
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[jnp.ndarray]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class FlaxIdefics2Encoder(nn.Module):
    dtype: jnp.dtype = jnp.float32
    config: Idefics2VisionConfig

    def setup(self):
        config = self.config
        self.layers = [
            FlaxIdefics2EncoderLayer(config, name=str(i), dtype=self.dtype) for i in range(config.num_hidden_layers)
        ]

    # Ignore copy
    def __call__(
        self,
        inputs_embeds,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, FlaxBaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class FlaxIdefics2VisionTransformer(nn.Module):
    config: Idefics2VisionConfig

    def setup(self):
        config = self.config
        embed_dim = config.hidden_size
        self.embeddings = FlaxIdefics2VisionEmbeddings(config)
        self.encoder = FlaxIdefics2Encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def __call__(
        self,
        pixel_values: jnp.ndarray,
        patch_attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, FlaxBaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_size = self.config.patch_size
            patch_attention_mask = jnp.ones(
                (
                    batch_size,
                    pixel_values.size(2) // patch_size,
                    pixel_values.size(3) // patch_size,
                )
            )
            patch_attention_mask = jax.device_put(patch_attention_mask.astype(jnp.bool), device=pixel_values.device)

        hidden_states = self.embeddings(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)

        patch_attention_mask = patch_attention_mask.view(batch_size, -1)
        # The call to `_upad_input` in `_flash_attention_forward` is expensive
        # So when the `patch_attention_mask` is full of 1s (i.e. attending to the whole sequence),
        # avoiding passing the attention_mask, which is equivalent to attending to the full sequence
        if not jnp.any(~patch_attention_mask):
            patch_attention_mask = None

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=patch_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        if not return_dict:
            return (last_hidden_state,) + encoder_outputs[1:]

        return FlaxBaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# end of vision portion


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Idefics2
class FlaxIdefics2RMSNorm(nn.Module):
    hidden_size: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        """
        Idefics2RMSNorm is equivalent to T5LayerNorm
        """
        self.weight = self.param("weight", lambda _, shape: jnp.ones(shape), self.hidden_size)
        self.variance_epsilon = self.eps

    def __call__(self, hidden_states):
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        # use `jax.numpy.sqrt` as `jax.lax.rsqrt` does not match `torch.rsqrt`
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)

        return self.weight * jnp.asarray(hidden_states, dtype=self.dtype)


class FlaxIdefics2PerceiverAttention(nn.Module):
    # TODO(czz): Idefics2PerceiverAttention
    pass


class FlaxIdefics2PerceiverLayer(nn.Module):
    config: Idefics2Config
    layer_idx: int

    def setup(self):
        config = self.config
        self.hidden_size = config.text_config.hidden_size
        self.n_latents = config.perceiver_config.resampler_n_latents
        self.depth = config.perceiver_config.resampler_depth
        self.rms_norm_eps = config.text_config.rms_norm_eps

        self.input_latents_norm = FlaxIdefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.input_context_norm = FlaxIdefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.self_attn = FlaxIdefics2PerceiverAttention(config, layer_idx=self.layer_idx)
        self.post_attention_layernorm = FlaxIdefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.mlp = FlaxIdefics2MLP(
            hidden_size=config.text_config.hidden_size,
            intermediate_size=config.text_config.hidden_size * 4,
            output_size=config.text_config.hidden_size,
            hidden_act=config.perceiver_config.hidden_act,
        )

    def __call__(
        self,
        latents: jnp.ndarray,
        context: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[jnp.ndarray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]]:
        residual = latents

        latents = self.input_latents_norm(latents)
        context = self.input_context_norm(context)

        latents, self_attn_weights, present_key_value = self.self_attn(
            latents=latents,
            context=context,
            attention_mask=attention_mask,
        )
        latents = residual + latents
        residual = latents

        latents = self.post_attention_layernorm(latents)
        latents = self.mlp(latents)
        latents = residual + latents

        outputs = (latents,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class FlaxIdefics2PerceiverResampler(nn.Module):
    config: Idefics2Config

    def setup(self) -> None:
        config = self.config
        self.hidden_size = config.text_config.hidden_size
        self.hidden_act = config.perceiver_config.hidden_act
        self.n_latents = config.perceiver_config.resampler_n_latents
        self.depth = config.perceiver_config.resampler_depth
        self.rms_norm_eps = config.text_config.rms_norm_eps

        # Create Latents for Perceiver
        self.latents = self.param("latents", lambda _, shape: jnp.ones(shape), (self.n_latents, self.hidden_size))

        # Create Transformer Blocks
        self.layers = [FlaxIdefics2PerceiverLayer(config, str(idx)) for idx in range(self.depth)]
        self.norm = FlaxIdefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

    def forward(
        self,
        context: jnp.ndarray,
        attention_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        # seq embed -> bsz seq embed

        # latents = self.latents.unsqueeze(0).expand((context.shape[0], *self.latents.size()))
        latents = self.latents
        latents = jnp.broadcast_to(jnp.expand_dims(latents, axis=0), (context.shape[0],) + latents.shape)

        latent_attention_mask = jnp.ones(
            (attention_mask.size(0), latents.size(1)), dtype=attention_mask.dtype, device=attention_mask.device
        )

        attention_mask = jnp.concat([attention_mask, latent_attention_mask], dim=-1)

        compressed_context = latents
        for perceiver_layer in self.layers:
            layer_outputs = perceiver_layer(
                compressed_context,
                context,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )

            compressed_context = layer_outputs[0]

        compressed_context = self.norm(compressed_context)

        return compressed_context


class FlaxIdefics2Connector(nn.Module):
    config: Idefics2Config

    def setup(self):
        config = self.config
        self.modality_projection = FlaxIdefics2MLP(
            hidden_size=config.vision_config.hidden_size,
            intermediate_size=config.text_config.intermediate_size,
            output_size=config.text_config.hidden_size,
            hidden_act=config.text_config.hidden_act,
        )
        self.perceiver_resampler = FlaxIdefics2PerceiverResampler(config)

    def forward(self, image_hidden_states, attention_mask):
        image_hidden_states = self.modality_projection(image_hidden_states)
        image_hidden_states = self.perceiver_resampler(context=image_hidden_states, attention_mask=attention_mask)
        return image_hidden_states


class FlaxIdefics2PreTrainedModel(FlaxPreTrainedModel):
    config_class = Idefics2Config
    base_model_prefix = "model"

    # _no_split_modules = ["Idefics2VisionAttention", "Idefics2MLP", "Idefics2PerceiverLayer", "Idefics2DecoderLayer"]
    # _skip_keys_device_placement = "past_key_values"
    # _supports_flash_attn_2 = True
    module_class: nn.Module = None

    def __init__(
        self,
        config: Idefics2Config,
        input_shape: Optional[Tuple[int]] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # overwriting _init_weights
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    # copied from FlaxMistralPreTrainedModel
    def init_cache(self, batch_size, max_length):
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return unfreeze(init_variables["cache"])
