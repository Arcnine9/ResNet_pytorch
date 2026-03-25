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
        return None
    
    def infer_output_shape(self, input_shape: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
        """根据layer_type和input_shape推断输出shape（简化版）"""
        if not input_shape:
            return None
            
        # 从name中解析shape（如果存在）
        match = re.search(r'\(([^)]+)\)', self.name)
        if match:
            try:
                dims = [int(x.strip()) for x in match.group(1).split(',')]
                return tuple(dims)
            except:
                pass
        return None


class LayerConfigParser:
    """解析layer.config文件"""
    
    # 支持的分隔符模式
    SEPARATORS = [
        '_' * 80,  # 80个下划线
        '_' * 20,  # 20个下划线（你的格式）
        '---' * 20,
        '===' * 20,
    ]
    
    @staticmethod
    def parse(layer_config_path: str) -> Dict[int, LayerInfo]:
        """
        解析layer.config文件，返回hook_id到LayerInfo的映射
        
        期望的输入格式（基于Hook ID文件）：
        Hook ID:0; Name:Conv2d (128,3,299,299)
        Next Layers:
        Next Layer 0 Hook ID:0; Name:BatchNorm2d (128,32,149,149)
        Input Tensor: tensor0 Is weight (global)?: 0, Size in byte: 137322496, Range:0--137322496
        ...
        ______________________________________________________________________________
        """
        if not os.path.exists(layer_config_path):
            raise FileNotFoundError(f"Layer config not found: {layer_config_path}")
            
        with open(layer_config_path, 'r') as f:
            content = f.read()
        
        # 尝试不同的分隔符
        blocks = []
        for sep in LayerConfigParser.SEPARATORS:
            if sep in content:
                blocks = [b.strip() for b in content.split(sep) if b.strip()]
                if len(blocks) > 1:
                    break
        
        # 如果没有找到分隔符，尝试按"Hook ID:"分割
        if not blocks:
            lines = content.split('\n')
            current_block = []
            for line in lines:
                if line.startswith('Hook ID:') and current_block:
                    blocks.append('\n'.join(current_block))
                    current_block = [line]
                else:
                    current_block.append(line)
            if current_block:
                blocks.append('\n'.join(current_block))
        
        layers = {}
        for block in blocks:
            layer_info = LayerConfigParser._parse_layer_block(block)
            if layer_info:
                layers[layer_info.hook_id] = layer_info
        
        print(f"[LayerConfigParser] Parsed {len(layers)} layers from {layer_config_path}")
        return layers
    
    @staticmethod
    def _parse_layer_block(block: str) -> Optional[LayerInfo]:
        """解析单个层的信息块"""
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if not lines:
            return None
            
        # 解析第一行：Hook ID和Name
        first_line = lines[0]
        
        # 支持多种格式：
        # "Hook ID:0; Name:Conv2d (128,3,299,299)"
        # "Layer 0: Conv2d (128,3,299,299)"
        hook_match = re.search(r'Hook ID[:\s]*(\d+)', first_line, re.IGNORECASE)
        if not hook_match:
            # 尝试其他格式
            hook_match = re.search(r'Layer\s*(\d+)', first_line, re.IGNORECASE)
        
        if not hook_match:
            return None
            
        hook_id = int(hook_match.group(1))
        
        # 提取Name
        name_match = re.search(r'Name[:\s]*(.+)', first_line)
        if name_match:
            full_name = name_match.group(1).strip()
        else:
            # 尝试从整行提取
            full_name = first_line.split(':', 1)[-1].strip() if ':' in first_line else first_line
        
        # 解析layer_type和shape
        layer_type, shape_info = LayerConfigParser._parse_name(full_name)
        
        # 初始化容器
        input_tensors = []
        output_tensors = []
        weight_tensors = []
        prev_layers = []
        next_layers = []
        
        section = None
        for line in lines[1:]:
            lower_line = line.lower()
            
            if 'next' in lower_line and 'layer' in lower_line and ':' in line:
                section = 'next'
                continue
            elif 'previous' in lower_line and 'layer' in lower_line and ':' in line:
                section = 'prev'
                continue
            elif line.startswith('Next Layer') or line.startswith('Previous Layer'):
                match = re.search(r'Hook ID[:\s]*(\d+)', line)
                if match:
                    layer_id = int(match.group(1))
                    if section == 'next':
                        next_layers.append(layer_id)
                    elif section == 'prev':
                        prev_layers.append(layer_id)
            elif 'tensor' in lower_line:
                tensor = LayerConfigParser._parse_tensor(line)
                if tensor:
                    if 'input' in lower_line and 'd_' not in line.lower():
                        input_tensors.append(tensor)
                    elif 'output' in lower_line and 'd_' not in line.lower():
                        output_tensors.append(tensor)
                    elif 'weight' in lower_line:
                        weight_tensors.append(tensor)
        
        return LayerInfo(
            hook_id=hook_id,
            name=full_name,
            layer_type=layer_type,
            shape_info=shape_info,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            weight_tensors=weight_tensors,
            prev_layers=prev_layers,
            next_layers=next_layers
        )
    
    @staticmethod
    def _parse_name(full_name: str) -> Tuple[str, Optional[Tuple[int, ...]]]:
        """解析名称，提取类型和shape"""
        # 匹配: Conv2d (128,3,299,299) 或 Conv2d((128,3,299,299))
        match = re.match(r'(\w+)\s*\(?\s*\(([^)]+)\)\s*\)?', full_name)
        if match:
            layer_type = match.group(1)
            shape_str = match.group(2)
            try:
                shape = tuple(int(x.strip()) for x in shape_str.split(','))
                return layer_type, shape
            except:
                return layer_type, None
        
        # 如果没有shape，只返回类型
        return full_name.split()[0] if full_name else "Unknown", None
    
    @staticmethod
    def _parse_tensor(line: str) -> Optional[Dict[str, Any]]:
        """解析tensor信息行"""
        # 支持多种格式
        patterns = [
            # 标准格式: Input Tensor: tensor0 Is weight (global)?: 0, Size in byte: 137322496
            r'(\w+)\s*Tensor:\s*(\w+).*?(?:Is weight.*?)?(\d+).*?Size.*?byte:\s*(\d+)',
            # 简化格式: tensor0: size=137322496
            r'(\w+):\s*size[=:]\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 4:
                    return {
                        'tensor_name': groups[1],
                        'is_weight': groups[2] == '1' or 'weight' in line.lower(),
                        'size': int(groups[3])
                    }
                elif len(groups) >= 2:
                    return {
                        'tensor_name': groups[0],
                        'is_weight': 'weight' in line.lower(),
                        'size': int(groups[1])
                    }
        return None


if __name__ == '__main__':
    # 测试代码
    import tempfile
    
    test_config = """Hook ID:0; Name:Conv2d (128,3,299,299)
Next Layers:
Next Layer 0 Hook ID:1; Name:BatchNorm2d (128,32,149,149)
Previous Layers:
Input Tensor: tensor0 Is weight (global)?: 0, Size in byte: 137322496, Range:0--137322496
Output Tensor: tensor1 Is weight (global)?: 0, Size in byte: 363741184, Range:137322496--501063680
______________________________________________________________________________
Hook ID:1; Name:BatchNorm2d (128,32,149,149)
Next Layers:
Previous Layers:
Previous Layer 0 Hook ID:0; Name:Conv2d (128,3,299,299)
Input Tensor: tensor1 Is weight (global)?: 0, Size in byte: 363741184, Range:137322496--501063680
______________________________________________________________________________"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.config', delete=False) as f:
        f.write(test_config)
        temp_path = f.name
    
    try:
        layers = LayerConfigParser.parse(temp_path)
        print(f"\n测试成功！解析了 {len(layers)} 层")
        for hook_id, info in layers.items():
            print(f"  Hook {hook_id}: {info.layer_type}, shape={info.shape_info}")
            print(f"    Input tensors: {len(info.input_tensors)}")
            print(f"    Prev layers: {info.prev_layers}")
    finally:
        os.unlink(temp_path)