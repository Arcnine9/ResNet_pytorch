"""
layer_config_parser.py
专职：解析离线分析器生成的 layer.config 文件
提取：层类型、shape、张量信息、连接关系
"""

import os
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class LayerInfo:
    """从layer.config解析的层信息"""
    hook_id: int
    name: str  # 如 "Conv2d (128,3,299,299)"
    layer_type: str  # 如 "Conv2d"
    shape_info: Tuple[int, ...]  # 如 (128, 3, 299, 299)
    
    # 张量信息
    input_tensors: List[Dict[str, Any]] = field(default_factory=list)
    output_tensors: List[Dict[str, Any]] = field(default_factory=list)
    weight_tensors: List[Dict[str, Any]] = field(default_factory=list)
    
    # 连接关系
    prev_layers: List[int] = field(default_factory=list)
    next_layers: List[int] = field(default_factory=list)
    
    def get_batch_size(self) -> Optional[int]:
        """从shape_info获取batch size"""
        if self.shape_info and len(self.shape_info) > 0:
            return self.shape_info[0]
        return None
    
    def get_input_shape(self) -> Optional[Tuple[int, ...]]:
        """从shape_info或input_tensors推断输入shape"""
        if self.shape_info:
            return self.shape_info
        # 从输入张量推断（简化）
        if self.input_tensors:
            first = self.input_tensors[0]
            if 'size' in first and first['size'] > 0:
                # 这里简化处理，实际应根据dtype和shape计算
                pass
        return None
    
    def infer_output_shape(self, input_shape: Optional[Tuple[int, ...]] = None) -> Optional[Tuple[int, ...]]:
        """
        根据layer_type和输入shape推断输出shape
        """
        if self.shape_info:
            return self.shape_info
        
        if not input_shape:
            return None
        
        layer_type_lower = self.layer_type.lower()
        
        # 卷积层
        if 'conv2d' in layer_type_lower:
            return input_shape  # 保守返回，实际应计算
            
        # 池化层
        elif 'maxpool' in layer_type_lower or 'avgpool' in layer_type_lower:
            if len(input_shape) >= 3:
                return input_shape[:-2] + (input_shape[-2]//2, input_shape[-1]//2)
            return input_shape
            
        # BatchNorm/ReLU/Dropout - shape不变
        elif any(x in layer_type_lower for x in ['batchnorm', 'relu', 'dropout']):
            return input_shape
            
        # Linear层
        elif 'linear' in layer_type_lower:
            if self.shape_info:
                return self.shape_info
            return input_shape
            
        return input_shape


class LayerConfigParser:
    """解析layer.config文件"""
    
    SEPARATORS = [
        '_' * 80,
        '_' * 78,
        '_' * 76,
        '_' * 82,
    ]
    
    @staticmethod
    def parse(layer_config_path: str) -> Dict[int, LayerInfo]:
        """解析layer.config文件"""
        if not os.path.exists(layer_config_path):
            raise FileNotFoundError(f"Layer config not found: {layer_config_path}")
            
        with open(layer_config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        blocks = []
        for sep in LayerConfigParser.SEPARATORS:
            if sep in content:
                potential_blocks = [b.strip() for b in content.split(sep) if b.strip()]
                if len(potential_blocks) > 1:
                    blocks = potential_blocks
                    break
        
        if not blocks:
            blocks = LayerConfigParser._split_by_hook_id(content)
        
        # 清理块
        cleaned_blocks = []
        for block in blocks:
            cleaned = block.lstrip('_ \n\r\t')
            if cleaned and 'Hook ID' in cleaned:
                cleaned_blocks.append(cleaned)
        
        layers = {}
        for block in cleaned_blocks:
            layer_info = LayerConfigParser._parse_layer_block(block)
            if layer_info:
                layers[layer_info.hook_id] = layer_info
        
        return layers
    
    @staticmethod
    def _split_by_hook_id(content: str) -> List[str]:
        """按Hook ID分割"""
        lines = content.split('\n')
        blocks = []
        current_block = []
        
        for line in lines:
            if re.match(r'^\s*Hook ID\s*:\s*\d+', line, re.IGNORECASE):
                if current_block:
                    blocks.append('\n'.join(current_block))
                    current_block = []
                current_block.append(line)
            else:
                if current_block:
                    current_block.append(line)
        
        if current_block:
            blocks.append('\n'.join(current_block))
        
        return blocks
    
    @staticmethod
    def _parse_layer_block(block: str) -> Optional[LayerInfo]:
        """解析单个层"""
        lines = block.split('\n')
        if not lines:
            return None
        
        first_line = ""
        for line in lines:
            stripped = line.strip()
            if stripped:
                first_line = stripped
                break
        
        if not first_line:
            return None
            
        hook_match = re.search(r'Hook ID\s*:\s*(\d+)', first_line, re.IGNORECASE)
        if not hook_match:
            return None
            
        hook_id = int(hook_match.group(1))
        
        full_name = ""
        name_match = re.search(r'Name\s*:\s*(.+)', first_line)
        if name_match:
            full_name = name_match.group(1).strip()
        
        layer_type, shape_info = LayerConfigParser._parse_name(full_name)
        
        input_tensors = []
        output_tensors = []
        weight_tensors = []
        prev_layers = []
        next_layers = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            lower_line = line.lower()
            
            if not line or line.startswith('_'):
                i += 1
                continue
            
            if lower_line.startswith('next layers:'):
                i += 1
                while i < len(lines):
                    sub_line = lines[i].strip()
                    if not sub_line or sub_line.startswith('_'):
                        i += 1
                        continue
                    
                    if 'previous' in sub_line.lower() or 'tensor' in sub_line.lower():
                        break
                    
                    if sub_line.lower().startswith('next layer'):
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            hook_id_match = re.search(r'Hook ID\s*:\s*(\d+)', next_line, re.IGNORECASE)
                            if hook_id_match:
                                next_layers.append(int(hook_id_match.group(1)))
                                i += 2
                                continue
                    i += 1
                continue
            
            elif lower_line.startswith('previous layers:'):
                i += 1
                while i < len(lines):
                    sub_line = lines[i].strip()
                    if not sub_line or sub_line.startswith('_'):
                        i += 1
                        continue
                    
                    if 'tensor' in sub_line.lower() or 'next' in sub_line.lower():
                        break
                    
                    if sub_line.lower().startswith('previous layer'):
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            hook_id_match = re.search(r'Hook ID\s*:\s*(\d+)', next_line, re.IGNORECASE)
                            if hook_id_match:
                                prev_layers.append(int(hook_id_match.group(1)))
                                i += 2
                                continue
                    i += 1
                continue
            
            elif 'tensor' in lower_line and not line.startswith('d_'):
                if any(x in lower_line for x in ['input tensor', 'output tensor', 'weight tensor']):
                    tensor = LayerConfigParser._parse_tensor(line)
                    if tensor:
                        if 'input tensor' in lower_line:
                            input_tensors.append(tensor)
                        elif 'output tensor' in lower_line:
                            output_tensors.append(tensor)
                        elif 'weight tensor' in lower_line:
                            weight_tensors.append(tensor)
            
            i += 1
        
        return LayerInfo(
            hook_id=hook_id,
            name=full_name,
            layer_type=layer_type,
            shape_info=shape_info,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            weight_tensors=weight_tensors,
            prev_layers=list(set(prev_layers)),
            next_layers=list(set(next_layers))
        )
    
    @staticmethod
    def _parse_name(full_name: str) -> Tuple[str, Optional[Tuple[int, ...]]]:
        """解析名称，提取类型和shape"""
        if not full_name:
            return "Unknown", None
            
        match = re.match(r'(\w+)\s*\(([^)]+)\)', full_name)
        if match:
            layer_type = match.group(1)
            shape_str = match.group(2)
            try:
                shape = tuple(int(x.strip()) for x in shape_str.split(','))
                return layer_type, shape
            except ValueError:
                return layer_type, None
        
        parts = full_name.split()
        return parts[0] if parts else "Unknown", None
    
    @staticmethod
    def _parse_tensor(line: str) -> Optional[Dict[str, Any]]:
        """解析tensor信息"""
        name_match = re.search(r'(\w+)\s*Tensor:\s*(\w+)', line)
        if not name_match:
            return None
        
        tensor_name = name_match.group(2)
        
        size_match = re.search(r'Size.*?byte:\s*(\d+)', line, re.IGNORECASE)
        size = int(size_match.group(1)) if size_match else 0
        
        is_weight = 'Is weight (global)?: 1' in line or 'Is weight (global)?:1' in line
        
        range_match = re.search(r'Range:(\d+)--(\d+)', line)
        tensor_range = None
        if range_match:
            tensor_range = (int(range_match.group(1)), int(range_match.group(2)))
        
        return {
            'tensor_name': tensor_name,
            'is_weight': is_weight,
            'size': size,
            'range': tensor_range
        }


if __name__ == '__main__':
    import sys
    
    test_file = "./layers.config"
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    
    if os.path.exists(test_file):
        parser = LayerConfigParser()
        layers = parser.parse(test_file)
        print(f"解析完成！共 {len(layers)} 层")
        
        for hid in sorted(layers.keys())[:5]:
            info = layers[hid]
            shape = info.get_input_shape()
            print(f"Hook {hid}: {info.layer_type} {shape}")
    else:
        print(f"文件 {test_file} 不存在")