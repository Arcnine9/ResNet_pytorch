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
        # 从输入张量推断
        if self.input_tensors:
            # 尝试从第一个输入张量获取
            first = self.input_tensors[0]
            if 'size' in first and first['size'] > 0:
                # 这需要知道dtype大小来反推shape，暂时返回None
                pass
        return None
    
    def infer_output_shape(self, input_shape: Optional[Tuple[int, ...]] = None) -> Optional[Tuple[int, ...]]:
        """
        根据layer_type和输入shape推断输出shape
        这是PreProfiler构造输入数据的关键方法
        """
        # 如果有明确的shape_info（从config解析的），优先使用
        if self.shape_info:
            # 但如果是动态层（如Linear, Flatten等），可能需要根据输入调整
            return self.shape_info
        
        if not input_shape:
            return None
        
        # 根据层类型计算输出shape
        layer_type_lower = self.layer_type.lower()
        
        # 卷积层 - 简化计算，假设stride=1, padding=0或根据config中的shape
        if 'conv2d' in layer_type_lower:
            # 如果有weight_tensors，可以从权重推断输出channel
            out_channels = None
            if self.weight_tensors:
                # 权重格式通常是 [out_ch, in_ch, kH, kW]
                weight_size = self.weight_tensors[0].get('size', 0)
                # 这里简化处理，假设已知out_channels
                pass
            # 如果config中有shape信息，使用它
            if self.shape_info:
                return self.shape_info
            return input_shape  # 保守返回，实际应该计算
            
        # 池化层 - 通常H,W减半，C不变
        elif 'maxpool' in layer_type_lower or 'avgpool' in layer_type_lower or 'adaptiveavgpool' in layer_type_lower:
            # 从name中解析目标shape，如 AdaptiveAvgPool2d (128,2048,1,1)
            if self.shape_info:
                return self.shape_info
            # 默认池化后H,W减半
            if len(input_shape) >= 3:
                return input_shape[:-2] + (input_shape[-2]//2, input_shape[-1]//2)
            return input_shape
            
        # BatchNorm - shape不变
        elif 'batchnorm' in layer_type_lower:
            return input_shape
            
        # ReLU - shape不变
        elif 'relu' in layer_type_lower:
            return input_shape
            
        # Dropout - shape不变
        elif 'dropout' in layer_type_lower:
            return input_shape
            
        # Linear/全连接层 - 从name中解析，如 Linear (128,1000)
        elif 'linear' in layer_type_lower:
            if self.shape_info:
                return self.shape_info
            # Linear通常将特征维度改变，但保持batch
            if len(input_shape) >= 1:
                return (input_shape[0],)  # 简化，实际需要知道out_features
            
        # Concat - 在指定维度合并，shape会改变
        elif 'concat' in layer_type_lower:
            if self.shape_info:
                return self.shape_info
            return input_shape  # 保守返回
            
        # 默认：返回输入shape（假设大部分层保持shape不变）
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
        
        # 找到第一个非空行
        first_line = ""
        for line in lines:
            stripped = line.strip()
            if stripped:
                first_line = stripped
                break
        
        if not first_line:
            return None
            
        # 解析 Hook ID
        hook_match = re.search(r'Hook ID\s*:\s*(\d+)', first_line, re.IGNORECASE)
        if not hook_match:
            return None
            
        hook_id = int(hook_match.group(1))
        
        # 提取 Name
        full_name = ""
        name_match = re.search(r'Name\s*:\s*(.+)', first_line)
        if name_match:
            full_name = name_match.group(1).strip()
        
        # 解析 layer_type 和 shape
        layer_type, shape_info = LayerConfigParser._parse_name(full_name)
        
        # 初始化
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
            
            # Next Layers 解析
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
            
            # Previous Layers 解析
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
            
            # 张量解析 - 只解析普通张量，跳过梯度(d_)张量
            elif 'tensor' in lower_line and not line.startswith('d_'):
                # 只解析 Input, Output, Weight（非梯度）
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
            
        # 匹配: Conv2d (128,3,299,299)
        match = re.match(r'(\w+)\s*\(([^)]+)\)', full_name)
        if match:
            layer_type = match.group(1)
            shape_str = match.group(2)
            try:
                shape = tuple(int(x.strip()) for x in shape_str.split(','))
                return layer_type, shape
            except ValueError:
                return layer_type, None
        
        # 备用
        parts = full_name.split()
        return parts[0] if parts else "Unknown", None
    
    @staticmethod
    def _parse_tensor(line: str) -> Optional[Dict[str, Any]]:
        """解析tensor信息"""
        # 匹配: Input Tensor: tensor0 Is weight (global)?: 0, Size in byte: 137322496, Range:0--137322496
        name_match = re.search(r'(\w+)\s*Tensor:\s*(\w+)', line)
        if not name_match:
            return None
        
        tensor_name = name_match.group(2)
        
        # 提取大小
        size_match = re.search(r'Size.*?byte:\s*(\d+)', line, re.IGNORECASE)
        size = int(size_match.group(1)) if size_match else 0
        
        # 判断是否为权重
        is_weight = 'Is weight (global)?: 1' in line or 'Is weight (global)?:1' in line
        
        # 提取范围
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
        
        # 验证 infer_output_shape 方法存在
        for hid in list(layers.keys())[:3]:
            info = layers[hid]
            shape = info.infer_output_shape(info.get_input_shape())
            print(f"Hook {hid}: {info.layer_type} -> shape {shape}")
    else:
        print(f"文件 {test_file} 不存在")