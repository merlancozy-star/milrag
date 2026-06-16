"""pipeline/serve.py — 部署服务（论文 6.2 三层架构 / 6.3.4 / 6.3.5）。

三层：数据层（KB/嵌入/FAISS+ES/PG）→ 控制层（vLLM/检测器/重构/循环）→ 防御层。

服务栈：
  - FastAPI：主 API 服务（/ask /health /trace）
  - gRPC：防御模块微服务（可独立扩缩）
  - Streamlit：前端（置信度标签/证据溯源/冲突警告/反馈）

完全离线内网部署。P95 延迟预算：基础事实 ≤3s，复杂推理 ≤10s（论文 6.4）。
"""
from __future__ import annotations


def build_fastapi_app(cfg: dict, orchestrator) -> "FastAPI":
    """构建 FastAPI 应用。

    Endpoints:
      POST /ask      — 问答（返回 answer + evidence + confidence + trace）
      GET  /health   — 健康检查
      GET  /trace/{id} — 获取单次问答的完整溯源
    """
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("请安装 fastapi: pip install fastapi uvicorn")

    app = FastAPI(title="milrag — Military Intelligence RAG",
                  version="1.0.0")

    # 内存中的 trace 存储（生产环境应置换到 PG）
    trace_store: dict[str, dict] = {}
    _counter = [0]

    class AskRequest(BaseModel):
        question: str
        return_trace: bool = False

    class AskResponse(BaseModel):
        answer: str
        evidence_ids: list[str]
        confidence: float
        refused: bool
        reason: str
        source: str
        latency_ms: float
        trace_id: str | None = None

    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest):
        result = orchestrator.answer(req.question)

        trace_id = None
        if req.return_trace and result.trace:
            _counter[0] += 1
            trace_id = f"trace_{_counter[0]:06d}"
            trace_store[trace_id] = result.trace

        return AskResponse(
            answer=result.answer,
            evidence_ids=[e.get("chunk_id", "") for e in result.evidence],
            confidence=result.confidence,
            refused=result.refused,
            reason=result.reason,
            source=result.source,
            latency_ms=result.latency_ms,
            trace_id=trace_id,
        )

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": cfg.get("paths", {}).get("models", {}).get("llm_backbone", "")}

    @app.get("/trace/{trace_id}")
    async def get_trace(trace_id: str):
        if trace_id not in trace_store:
            return {"error": "trace not found"}
        return {"trace_id": trace_id, "steps": trace_store[trace_id]}

    return app


def build_streamlit_app(cfg: dict, api_url: str = "http://localhost:8000"):
    """构建 Streamlit 前端（论文 6.3.5）。

    功能：
      - 问答输入框
      - 置信度彩色标签（红/黄/绿）
      - 证据溯源面板（折叠显示每条证据的来源/时间/权威性）
      - 冲突警告横幅
      - 用户反馈按钮（有用/无用/有害）
    """
    try:
        import streamlit as st
    except ImportError:
        raise ImportError("请安装 streamlit: pip install streamlit")

    st.set_page_config(page_title="军事情报问答系统", layout="wide")

    st.title("🔍 军事情报动态鲁棒 RAG 问答系统")
    st.caption("基于多信号动态检索与两阶段可信增强 — 完全离线内网部署")

    question = st.text_input("请输入您的军事情报问题：",
                             placeholder="例如：歼-20 的作战半径为多少公里？")

    if question:
        import requests
        with st.spinner("正在检索和分析..."):
            try:
                resp = requests.post(
                    f"{api_url}/ask",
                    json={"question": question, "return_trace": True},
                    timeout=30,
                )
                data = resp.json()
            except Exception as e:
                st.error(f"服务连接失败：{e}")
                return

        # 答案
        st.markdown("### 📋 回答")
        if data["refused"]:
            st.warning(f"⚠️ 系统拒答：{data['reason']}")
        else:
            st.info(data["answer"])

        # 置信度标签
        conf = data["confidence"]
        color = "green" if conf > 0.7 else ("orange" if conf > 0.4 else "red")
        st.markdown(f"**置信度**: :{color}[{conf:.2%}]")

        # 证据溯源
        if data.get("evidence_ids"):
            with st.expander("📚 证据溯源",
                             expanded=False):
                for eid in data["evidence_ids"]:
                    st.text(f"• {eid}")

        # 延迟
        st.caption(f"⏱️ 延迟: {data['latency_ms']:.0f}ms | 来源: {data['source']}")

        # 反馈
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("👍 有用")
        with col2:
            st.button("👎 无用")
        with col3:
            st.button("⚠️ 有害/误导")


def build_app(cfg: dict, orchestrator=None) -> "FastAPI":
    """一站式构建 FastAPI 应用。

    Args:
        cfg: config/base.yaml（含 serve 相关配置）。
        orchestrator: Orchestrator 实例（可选，若为 None 则只启动 API 框架）。

    Returns:
        FastAPI app 实例。
    """
    if orchestrator is None:
        # 空壳模式：只返回 app 框架供测试
        from fastapi import FastAPI
        app = FastAPI(title="milrag")
        return app

    return build_fastapi_app(cfg, orchestrator)
