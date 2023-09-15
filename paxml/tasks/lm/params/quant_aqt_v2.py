"""Wrapper classes for enabling quantization during training."""
from absl import flags
from absl import logging
from aqt.jax.v2 import aqt_dot_general
from aqt.jax.v2 import config
from fiddle import selectors
import jax.numpy as jnp
from praxis import base_layer
from praxis import layers
from praxis import pax_fiddle
from praxis.layers import transformers


_MLPERF_GPT3_AQT_CONFIG_NAME = flags.DEFINE_enum(
    'mlperf_gpt3_aqt_config_name',
    default='ttf_quant_v1',
    enum_values=['ttf_quant_v1', 'fully_quantized_v0'],
    help='accumulator dtype',
)


def get_aqt_config(
        aqt_cfg_name: str = _MLPERF_GPT3_AQT_CONFIG_NAME.value,
        ) -> config.DotGeneral:
  """return aqt config."""

  if aqt_cfg_name == 'ttf_quant_v1':
    fwd = config.DotGeneralRaw.make(8, 8)
    dlhs = config.DotGeneralRaw.make(8, 8)
    drhs = config.DotGeneralRaw.make(None, None)
    aqt_cfg = config.DotGeneral(fwd=fwd, dlhs=dlhs, drhs=drhs)

    # Surprising: lhs quantization determines what drhs can do.
    # Only rhs is accepting MultiTensor.
    aqt_cfg.drhs.rhs.use_fwd_quant = False
    aqt_cfg.dlhs.rhs.use_fwd_quant = False
    config.set_stochastic_rounding(
        aqt_cfg,
        vjp_lhs_stochastic_rounding=True,
        vjp_rhs_stochastic_rounding=False,
        implementation='custom-1'
    )
    logging.info('>>> AQT(int8_ttf_quant_v1) config: %s', aqt_cfg)

  elif aqt_cfg_name == 'fully_quantized_v0':
    aqt_cfg = config.fully_quantized(
        fwd_bits=8,
        bwd_bits=8,
        use_fwd_quant=False,
        use_stochastic_rounding=None,
        vjp_lhs_stochastic_rounding=True,
        vjp_rhs_stochastic_rounding=False,
        use_dummy_static_bound=False,
    )
    config.set_stochastic_rounding(
        aqt_cfg,
        vjp_lhs_stochastic_rounding=True,
        vjp_rhs_stochastic_rounding=False,
        implementation='custom-1'
    )

    accumulator_dtype = jnp.int32
    config.set_accumulator_dtype(
        aqt_cfg,
        fwd_dtype=accumulator_dtype,
        bwd_dtype=accumulator_dtype,
    )

    logging.info('>>> AQT config(fully_quantized_v0): %s', aqt_cfg)
  else:
    raise ValueError(f'Unsupported aqt_cfg_name: {aqt_cfg_name}')
  return aqt_cfg


class DqQuantEinsum(base_layer.BaseLayer):
  """Einsum layer with quantization."""

  def __call__(self, eq, lhs, rhs):
    aqt_key = self.next_prng_key()
    def dg(lhs, rhs, axes, precision=None, preferred_element_type=None):
      del precision, preferred_element_type

      # Stochastic rounding in applied only to the gradient tensor i.e. lhs
      aqt_cfg = get_aqt_config(aqt_cfg_name=_MLPERF_GPT3_AQT_CONFIG_NAME.value)
      dot = aqt_dot_general.make_dot_general(aqt_cfg)
      context = aqt_dot_general.Context(key=aqt_key, train_step=None)

      return dot(lhs, rhs, axes, context=context)

    return jnp.einsum(eq, lhs, rhs, _dot_general=dg)


def apply_quantized_layers_sharded(model):
  """Adds quantization to existing transformer layers."""
  einsum_tpl = pax_fiddle.Config(DqQuantEinsum)

  if hasattr(model, 'lm_tpl'):
    logging.info('quantize attention: QKV')
    # Quantize attention: QKV
    selectors.select(model, layers.attentions.CombinedQKVProjectionLayer).set(
        einsum_tpl=einsum_tpl
    )
    logging.info('quantize attention projection')
    # Quantize attention projection
    selectors.select(model, layers.attentions.AttentionProjection).set(
        einsum_tpl=einsum_tpl
    )
    logging.info('quantize attention output projection')
    # Quantize attention output projection
    selectors.select(model, layers.attentions.DotProductAttention).set(
        qk_einsum_tpl=einsum_tpl,
        pv_einsum_tpl=einsum_tpl,
    )
    logging.info('quantize feedforward layers')
    # Quantize feedforward layers.
    xformer_p = model.lm_tpl.stacked_transformer_tpl
    if xformer_p.cls == transformers.PipelinedTransformer:
      xformer_p = xformer_p.pipeline_stage

    if xformer_p.cls == transformers.StackedTransformerRepeated:
      xformer_p = xformer_p.block
    xformer_p = xformer_p.transformer_layer_params_tpl
    xformer_p.tr_fflayer_tpl.fflayer_tpl.linear_tpl.set(einsum_tpl=einsum_tpl)
