# Machine Learning 练手项目

记录个人学习 ML/DL 的项目

---

## Projects

### 1. [股票预测](股票预测/) — 2026-02

基于 LSTM 的股票成交量趋势预测。使用 akshare 获取 A 股历史数据，滑动窗口构建特征，PyTorch 实现 LSTM 模型训练与预测。

- **来源**：《动手学PyTorch建模与应用：从深度学习到大模型》课后习题
- **涉及内容**: PyTorch, LSTM, akshare, pandas, matplotlib

### 2. [LLM Classification Finetuning](LLM%20Classification%20Finetuning/) — 2026-02

LLM 输出对比分类任务。对两个模型的 response 进行分类（A 胜 / B 胜 / 平局），尝试 TF-IDF + 逻辑回归 和 BERT 特征提取两种方案。

- **来源**：Kaggle 竞赛 + GitHub 参考 | [Kaggle](https://www.kaggle.com/competitions/llm-classification-finetuning) / [GitHub](https://github.com/Nawres2020/LLM-Classification-Finetuning)
- **涉及内容**: scikit-learn, HuggingFace Transformers, BERT, TF-IDF

### 3. [问答系统](问答系统/) — 2026-04

基于 RAG（检索增强生成）的智能问答系统。使用 LangChain + DashScope 向量模型 + 通义千问 LLM，支持文件上传、知识库管理、对话历史。

- **来源**：B站黑马程序员教程 | [链接](https://www.bilibili.com/video/BV1yjz5BLEoY/)
- **涉及内容**: LangChain, 通义千问, DashScope, Streamlit, ChromaDB

### 4. [table-detection](table-detection/) — 2026-05

基于 YOLO 的桌面物体检测。从 Roboflow 获取数据集，训练识别 11 类桌面物品，支持 ONNX 导出与实时推理。

- **来源**：YOLO 官方教程 | [链接](https://docs.ultralytics.com/zh)
- **涉及内容**: Ultralytics YOLO, OpenCV, ONNX

---

## 学习进度

| 时间 | 内容 |
|------|------|
| 2026-02 | 股票预测、LLM 分类微调 |
| 2026-04 | RAG 问答系统 |
| 2026-05 | YOLO 桌面物体检测 |
