# Ultralytics Multimodal Config Parser
# Universal YAML configuration parsing for RGB+X architectures  
# Version: v1.0

from ultralytics.utils import LOGGER
import re


class MultiModalConfigParser:
    """
    Universal Multimodal Configuration Parser
    
    Handles YAML configuration parsing for both YOLO and RTDETR
    with RGB+X multimodal extensions
    """
    
    def __init__(self):
        self.supported_input_sources = ['RGB', 'X', 'Dual']
    
    def validate_config_format(self, config):
        """Validate multimodal configuration format correctness"""
        
        rgb_layers = []
        x_layers = []
        dual_layers = []
        
        for section in ['backbone', 'head']:
            for i, layer_config in enumerate(config.get(section, [])):
                if len(layer_config) == 5:  # Has 5th field
                    input_source = layer_config[4]
                    if input_source == 'RGB':
                        rgb_layers.append(i)
                    elif input_source == 'X':
                        x_layers.append(i)
                    elif input_source == 'Dual':
                        dual_layers.append(i)
        
        LOGGER.info("✅ MultiModal: 配置验证完成")
        LOGGER.info(f"MultiModal: RGB路由层={len(rgb_layers)}, X路由层={len(x_layers)}, Dual路由层={len(dual_layers)}")
        
        return {
            'rgb_layers': rgb_layers,
            'x_layers': x_layers, 
            'dual_layers': dual_layers,
            'total_routing_layers': len(rgb_layers) + len(x_layers) + len(dual_layers)
        }
    
    def extract_multimodal_info(self, config):
        """Extract multimodal information from configuration"""
        
        # Get X modality type from dataset config
        x_modality_type = config.get('dataset_config', {}).get('x_modality', 'unknown')
        
        # Count multimodal layers
        mm_layer_count = 0
        for section in ['backbone', 'head']:
            for layer_config in config.get(section, []):
                if len(layer_config) >= 5 and layer_config[4] in self.supported_input_sources:
                    mm_layer_count += 1
        
        return {
            'x_modality_type': x_modality_type,
            'mm_layer_count': mm_layer_count,
            'supports_multimodal': mm_layer_count > 0
        } 

    def parse_config(self, config: dict) -> dict:
        """
        Build a minimal multimodal model_config dict for MultiModalRouter.

        Detects whether YAML has any 5th-field multimodal routing markers.
        Returns original config with helper flags added; this function keeps
        backward-compatibility by not enforcing any schema beyond what's required
        by the current router implementation.
        """
        has_mm = False
        input_layers = []
        for section in ['backbone', 'head']:
            for i, layer_config in enumerate(config.get(section, [])):
                if len(layer_config) >= 5 and layer_config[4] in self.supported_input_sources:
                    has_mm = True
                    input_layers.append((section, i))
        out = dict(config)
        out['has_multimodal_layers'] = has_mm
        out['input_layers'] = input_layers
        return out

    # ------------------------------
    # Hook (6th field) parsing utils
    # ------------------------------
    def parse_hook_field(self, hook_field, layer_idx: int):
        """
        Parse YAML 6th-field hook DSL into a list of normalized specs.

        Expected forms:
            [CL, <modality>, <stage>, key=value, ...]
            or [[CL, ...], [CL, ...]] for multiple hooks on the same layer.

        Returns:
            list[dict]: Each dict contains keys:
                tool: 'CL'
                modality: 'RGB'|'X'
                stage: 'P3'|'P4'|'P5'|...
                tap: 'output'|'input' (default 'output')
                action: 'capture' (default 'capture')
                    NOTE: 'emit_sim' and 'capture_grad' are reserved keywords and NOT implemented.
                detach: bool (default False)
                normalize: bool (default False)
                buffer: Optional[str] (debug/visualization naming override)
                layer_idx: int (for auto-naming stability)

        Raises:
            ValueError: on invalid syntax/values.
        """
        if hook_field is None:
            return []

        # Normalize to list of hooks
        hooks = hook_field
        if not isinstance(hooks, list):
            raise ValueError(f"hook field must be a list, got: {type(hook_field)}")
        if len(hooks) == 0:
            return []
        # Single hook may be flat list like [CL, RGB, P4, ...]
        if hooks and isinstance(hooks[0], str):
            hooks = [hooks]

        out = []
        for h in hooks:
            if not isinstance(h, list) or len(h) < 3:
                raise ValueError(f"invalid hook spec (need at least 3 tokens): {h}")
            tool, modality, stage = h[0], h[1], h[2]
            if tool != 'CL':
                raise ValueError(f"unsupported hook tool: {tool}")
            if modality == 'Fused':
                modality = 'Dual'
            if modality not in ('RGB', 'X', 'Dual'):
                raise ValueError(f"unsupported modality in hook: {modality}")
            if not isinstance(stage, str) or not re.fullmatch(r"P\d+", stage):
                raise ValueError(f"invalid stage name: {stage}")

            # Defaults
            spec = {
                'tool': 'CL',
                'modality': modality,
                'stage': stage,
                'tap': 'output',
                'action': 'capture',
                'detach': False,
                'normalize': False,
                'layer_idx': int(layer_idx),
            }

            # Parse key=value tokens
            for token in h[3:]:
                if isinstance(token, str) and '=' in token:
                    k, v = token.split('=', 1)
                elif isinstance(token, (list, tuple)) and len(token) == 2:
                    k, v = token  # allow YAML pairs
                else:
                    raise ValueError(f"invalid key=value token in hook: {token}")

                k = str(k).strip()
                v_raw = v
                v = str(v).strip() if not isinstance(v, bool) else v

                if k == 'tap':
                    if v not in ('output', 'input'):
                        raise ValueError(f"tap must be 'output' or 'input', got {v}")
                    spec['tap'] = v
                elif k == 'action':
                    # Temporarily only support 'capture'. Other values are reserved but not implemented.
                    if v != 'capture':
                        raise ValueError(
                            f"unsupported action: {v} (reserved keyword; not implemented)"
                        )
                    spec['action'] = 'capture'
                elif k == 'buffer':
                    # Optional debug name override
                    spec['buffer'] = str(v)
                elif k == 'detach':
                    if isinstance(v_raw, bool):
                        spec['detach'] = v_raw
                    else:
                        spec['detach'] = v.lower() == 'true'
                elif k == 'normalize':
                    if isinstance(v_raw, bool):
                        spec['normalize'] = v_raw
                    else:
                        spec['normalize'] = v.lower() == 'true'
                else:
                    raise ValueError(f"unknown hook key: {k}")

            out.append(spec)

        LOGGER.debug(f"Parsed {len(out)} hook spec(s) at layer {layer_idx}")
        return out
