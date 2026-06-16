"""milrag.data — 数据清洗 / NER / 分块 / 索引 / QA构造。"""
from milrag.data.clean import clean_document, extract_references
from milrag.data.ner import MilitaryNER, Entity, normalize_entity, get_entity_positions
from milrag.data.chunk import chunk_document, Chunk
from milrag.data.index_build import build_faiss, load_faiss, build_es, init_metadata_db
from milrag.data.build_qa import generate_candidates, quality_checks, build_dataset
from milrag.data.build_adversarial import poison_sample, build_adversarial_dataset

__all__ = [
    "clean_document", "extract_references",
    "MilitaryNER", "Entity", "normalize_entity", "get_entity_positions",
    "chunk_document", "Chunk",
    "build_faiss", "load_faiss", "build_es", "init_metadata_db",
    "generate_candidates", "quality_checks", "build_dataset",
    "poison_sample", "build_adversarial_dataset",
]
