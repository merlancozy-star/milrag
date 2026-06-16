"""milrag.llm — LLM 推理后端（双后端）与白盒信号钩子。"""
from milrag.llm.backbone import Backbone
from milrag.llm.hooks import SignalCaptureProcessor, AttentionCaptureHook, register_attention_hook

__all__ = ["Backbone", "SignalCaptureProcessor", "AttentionCaptureHook", "register_attention_hook"]
