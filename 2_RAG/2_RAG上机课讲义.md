# 检索增强生成（RAG）上机课讲义

## 课程基本信息
- **课程名称**：大数据与商务智能 - 检索增强生成（RAG）系统实践
- **授课对象**：中山大学工商管理非全专硕MBA学生
- **课时**：3学时（上机课）
- **前置知识**：基础Python编程能力，LLM基础应用能力

## 学习目标
完成本课程后，您将能够：
1. 独立搭建基于Milvus和LlamaIndex的RAG系统，实现企业知识库检索
2. 掌握向量数据库的索引优化和性能调优方法，提升检索效率30%以上
3. 设计并实现混合检索（密集+稀疏向量）解决方案，提高查询准确率
4. 运用Cherry Studio可视化工具构建完整的企业知识库应用
5. 具备RAG系统部署和维护的基本能力，识别并解决常见问题

## 2.1 RAG环境配置

### 2.1.1 创建虚拟环境
```bash
python -m venv rag-env
```
# Windows: rag-env\Scripts\activate.bat
# macOS/Linux: source rag-env/bin/activate

```bash
pip install -U pip
pip install llama-index==0.10.18 pymilvus==2.4.0 python-dotenv==1.0.0 \
            PyPDF2==3.0.1 python-docx==1.0.0 streamlit==1.31.0 \
            pandas==2.2.0 scikit-learn==1.4.0
```


### 2.1.2 安装Milvus向量数据库
Windows: https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-windows-amd64.zip
macOS: https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-darwin-amd64.zip
Linux: https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-linux-amd64.tar.gz

### 2.1.3 解压并启动Milvus（以windows为例）

### 2.1.4 验证Milvus是否启动成功
# 查看日志确认服务状态
cat logs/standalone/milvus.log | grep "Milvus standalone started successfully"

### 2.1.5 安装Ollama与模型（如未安装）
下载地址：https://ollama.com/download

安装完成后下载模型
ollama pull llama3:8b


## 2.2 基础RAG系统构建与高级优化

### 2.2.1 文档加载与预处理
创建`1_basic_rag/document_loader.py`:
```python
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from typing import List
from datetime import datetime

def load_and_process_documents(
    data_dir: str = "../data",
    chunk_size: int = 512,
    chunk_overlap: int = 20
) -> List[Document]:
    """
    加载文档并进行分块处理（继承LLM课程的文档处理技术）
    """
    # 1. 加载文档
    print(f"从 {data_dir} 加载文档...")
    reader = SimpleDirectoryReader(
        input_dir=data_dir,
        recursive=True,
        required_exts=[".pdf", ".docx", ".txt"],
    )
    
    documents = reader.load_data()
    print(f"成功加载 {len(documents)} 个文档")
    
    # 2. 添加元数据（扩展LLM课程的元数据管理）
    for doc in documents:
        file_name = doc.metadata.get("file_name", "unknown_source")
        doc.metadata["source"] = file_name
        doc.metadata["processed_date"] = str(datetime.now().date())
        doc.metadata["course"] = "RAG"  # 新增课程标识
        
    # 3. 文档分块（优化LLM课程的分块策略）
    print(f"对文档进行分块处理 (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})")
    parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n\n"  # 优先按段落分割
    )
    
    nodes = parser.get_nodes_from_documents(documents)
    
    print(f"文档分块完成，共生成 {len(nodes)} 个文档块")
    print(f"示例块内容: {nodes[0].text[:100]}...")
    
    return nodes

if __name__ == "__main__":
    nodes = load_and_process_documents()
    
    # 保存处理结果
    import pickle
    with open("processed_nodes.pkl", "wb") as f:
        pickle.dump(nodes, f)
    print("文档处理结果已保存至 processed_nodes.pkl")
```

### 2.2.2 向量数据库集成
创建`1_basic_rag/vector_db_setup.py`:
```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import pickle
from datetime import datetime

def build_rag_index(
    nodes_path: str = "processed_nodes.pkl",
    collection_name: str = "basic_rag_collection",
    overwrite: bool = True
):
    """
    构建RAG系统索引（LLM课程向量应用的进阶实现）
    """
    # 1. 加载预处理文档
    print(f"从 {nodes_path} 加载文档块...")
    with open(nodes_path, "rb") as f:
        nodes = pickle.load(f)
    print(f"加载完成，共 {len(nodes)} 个文档块")
    
    # 2. 初始化嵌入模型（扩展LLM课程的模型应用）
    print("初始化嵌入模型...")
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5",
        embed_batch_size=16
    )
    
    # 3. 配置Milvus向量存储（RAG特有技术点）
    print(f"连接Milvus向量数据库，集合名称: {collection_name}")
    vector_store = MilvusVectorStore(
        uri="./milvus_rag.db",  # 本地文件存储
        collection_name=collection_name,
        dim=768,  # BGE模型输出维度
        overwrite=overwrite,
        similarity_metric="COSINE",
        index_config={
            "index_type": "HNSW",  # 高效近似最近邻索引
            "params": {"M": 16, "efConstruction": 256}
        }
    )
    
    # 4. 创建索引
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    print("RAG索引构建完成")
    return index

def create_query_engine(index):
    """创建查询引擎（继承LLM课程的查询设计）"""
    query_engine = index.as_query_engine(
        similarity_top_k=5,  # 返回前5个相关文档
        streaming=False
    )
    return query_engine

if __name__ == "__main__":
    index = build_rag_index()
    
    # 创建查询引擎
    query_engine = create_query_engine(index)
    
    # 测试查询（使用LLM课程相同的测试案例，便于对比）
    test_queries = [
        "企业去年的营收是多少？",
        "产品的核心功能有哪些？",
        "营销策略包含哪些渠道？"
    ]
    
    print("\n===== 测试RAG系统 =====对比LLM直出结果====")
    for query in test_queries:
        print(f"查询: {query}")
        response = query_engine.query(query)
        print(f"RAG增强回答: {str(response)[:200]}...\n")
    
    # 保存索引配置
    index.storage_context.persist(persist_dir="rag_index_storage")
    print("索引配置已保存至 rag_index_storage 目录")
```

### 2.2.3 高级RAG技术应用

#### 2.2.3.1 混合检索系统实现
```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.embeddings import SparseEmbedding
from typing import List, Dict, Any
import os
import pickle
from rank_bm25 import BM25Okapi

# 自定义BM25稀疏嵌入函数（RAG高级技术）
class BM25SparseEmbeddingFunction:
    def __init__(self):
        self.bm25 = None
        self.tokenizer = lambda x: x.split()
        self.corpus = []
        self.vocab = set()

    def fit(self, texts: List[str]):
        """训练BM25模型"""
        self.corpus = [self.tokenizer(text) for text in texts]
        self.bm25 = BM25Okapi(self.corpus)
        # 收集词汇表
        for doc in self.corpus:
            self.vocab.update(doc)
        return self

    def __call__(self, text: str) -> SparseEmbedding:
        """生成稀疏嵌入"""
        tokens = self.tokenizer(text)
        doc_scores = self.bm25.get_scores(tokens)
        
        # 创建稀疏向量（仅保留非零分数）
        indices = [i for i, score in enumerate(doc_scores) if score > 0]
        values = [float(score) for score in doc_scores if score > 0]
        
        return SparseEmbedding(indices=indices, values=values)

def build_hybrid_rag_system():
    """构建混合检索RAG系统（RAG进阶内容）"""
    # 1. 加载预处理文档
    with open("processed_nodes.pkl", "rb") as f:
        nodes = pickle.load(f)
    texts = [node.text for node in nodes]
    
    # 2. 初始化嵌入模型（融合LLM与检索技术）
    dense_embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    sparse_embed_model = BM25SparseEmbeddingFunction().fit(texts)
    
    # 3. 配置Milvus混合检索
    vector_store = MilvusVectorStore(
        uri="./milvus_hybrid.db",
        collection_name="hybrid_rag_collection",
        dim=768,
        overwrite=True,
        enable_sparse=True,
        sparse_embedding_function=sparse_embed_model,
        hybrid_ranker="WeightedRanker",
        hybrid_ranker_params={"weights": [0.7, 0.3]}  # 密集:稀疏 = 7:3
    )
    
    # 4. 创建索引
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        nodes,
        storage_context=storage_context,
        embed_model=dense_embed_model
    )
    
    print("混合检索RAG系统构建完成")
    return index
```

#### 2.2.3.2 性能优化策略
1. **异步查询实现**：
```python
async def batch_query(self, queries: list) -> list:
    """批量执行异步查询（RAG性能优化技术）"""
    tasks = [self.async_query(q) for q in queries]
    results = await asyncio.gather(*tasks)
    return results
```

2. **索引优化对比**：
```python
index_params_list = [
    {"index_type": "IVF_FLAT", "params": {"nlist": 128}},  # 基础索引
    {"index_type": "HNSW", "params": {"M": 16, "efConstruction": 256}}  # 优化索引
]
```

## 2.3 使用Cherry Studio构建知识库

### 2.3.1 Cherry Studio知识库功能概述

Cherry Studio提供了完整的知识库管理功能，支持从文档导入、向量化处理到检索优化的全流程可视化操作。根据[官方文档](https://docs.cherry-ai.com/knowledge-base/knowledge-base)，其核心功能包括：

- **多源文档导入**：支持PDF、Word、TXT等多种格式
- **智能分块处理**：自动优化文档分块策略
- **向量存储管理**：支持Milvus/FAISS等多种向量数据库
- **检索策略配置**：灵活调整检索参数和相似度阈值
- **知识库测试**：内置查询测试和结果分析工具

拓展资源：
- **Cherry Studio官方文档**：https://docs.cherry-ai.com/knowledge-base/knowledge-base
- **知识库最佳实践**：https://docs.cherry-ai.com/knowledge-base/best-practices
- **高级检索策略**：https://docs.cherry-ai.com/knowledge-base/advanced-retrieval
- **性能优化指南**：https://docs.cherry-ai.com/knowledge-base/performance-tuning

### 2.3.2 知识库创建与配置

#### 2.3.2.1 创建新知识库
1. 启动Cherry Studio，进入"知识库"模块
2. 点击"新建知识库"，填写基本信息：
   - 名称：企业知识库
   - 描述：用于存储和检索企业内部文档
   - 存储位置：本地
   - 向量数据库：Milvus（已在环境准备中配置）
3. 点击"创建"，系统将初始化知识库结构

#### 2.3.2.2 配置知识库参数
1. 进入知识库详情页，切换到"设置"标签
2. 配置文档处理参数：
   ```json
   {
     "chunk_size": 512,
     "chunk_overlap": 50,
     "separator": "\n\n",
     "include_metadata": true
   }
   ```
3. 配置嵌入模型：
   - 模型名称：BAAI/bge-base-en-v1.5
   - 批量大小：16
   - 设备：自动
4. 点击"保存配置"并应用

### 2.3.3 文档导入与管理

#### 2.3.3.1 导入文档
1. 进入"文档管理"标签，点击"导入文档"
2. 选择导入方式：
   - 方式1：本地上传（支持多文件批量上传）
   - 方式2：目录同步（监控指定目录自动导入）
3. 选择测试文档（可使用课程提供的sample_docs.zip）
4. 点击"开始导入"，系统将自动处理文档

#### 2.3.3.2 文档处理监控
1. 导入过程中，可在"任务中心"查看处理进度
2. 处理完成后，查看文档统计信息：
   - 总文档数：X个
   - 总文档块数：Y个
   - 平均分块大小：Z tokens
3. 检查是否有处理失败的文档，点击"重试"处理异常文档

### 2.3.4 检索策略配置

#### 2.3.4.1 基础检索配置
1. 进入"检索设置"标签，配置基础参数：
   - 相似度阈值：0.75
   - 返回结果数：5
   - 重排序：启用（基于BM25）
2. 点击"测试检索"，输入测试查询："企业核心产品有哪些？"
3. 查看检索结果和相关性评分，调整参数优化结果

#### 2.3.4.2 高级检索策略（混合检索配置）
1. 在"检索设置"中启用"混合检索"
2. 配置参数：
   - 密集向量权重：0.7
   - 稀疏向量权重：0.3
   - 交叉编码器重排序：启用
3. 点击"保存并应用"，系统将自动重建索引
4. 对比启用前后的检索效果差异

### 2.3.5 知识库应用开发

#### 2.3.5.1 创建检索应用
1. 进入"应用构建"模块，点击"新建应用"
2. 选择模板："知识库问答应用"
3. 配置应用参数：
   - 名称：企业知识库问答
   - 关联知识库：选择之前创建的"企业知识库"
   - 模型：llama3:8b（本地Ollama模型）
   - 提示词模板：使用RAG专用模板
4. 点击"创建应用"，系统自动生成应用界面

#### 2.3.5.2 应用测试与优化
1. 进入应用详情页，点击"预览"
2. 在测试界面输入问题，测试不同类型查询：
   - 事实型："企业成立时间？"
   - 概念型："什么是核心竞争力？"
   - 分析型："分析产品市场优势"
3. 查看回答质量和引用来源，记录需要优化的案例
4. 进入"优化中心"，针对低质量回答调整：
   - 修改分块参数
   - 优化提示词模板
   - 调整检索策略

### 2.3.6 知识库自动更新

1. 进入知识库"设置"→"自动更新"
2. 配置更新策略：
   - 更新频率：每周日凌晨2点
   - 更新范围：新增文档和修改过的文档
   - 通知方式：邮件通知
3. 点击"启用自动更新"

### 2.3.7 常见问题与解决方案

#### 文档导入失败
**问题**：PDF文档导入后内容为空  
**解决方案**：
1. 检查PDF是否加密或扫描件（需OCR处理）
2. 进入"设置"→"文档处理"，启用"高级PDF解析"
3. 重新导入文档

#### 检索结果相关性低
**问题**：查询结果与问题相关性差  
**解决方案**：
1. 降低相似度阈值（如从0.8调整至0.7）
2. 增加chunk_overlap至100
3. 尝试不同的嵌入模型（如切换至bge-large模型）

#### 知识库体积过大
**问题**：知识库文档过多导致检索缓慢  
**解决方案**：
1. 启用"分层检索"（设置知识库层级结构）
2. 配置"自动归档"策略，将旧文档移至归档库
3. 优化索引参数，使用IVF_FLAT索引类型

## 2.4 RAG提示词工程

### 2.4.1 检索增强提示结构
```markdown
# 系统角色
你是企业知识库问答专家，使用提供的检索结果回答问题。

## 检索结果使用规则
- 仅使用检索到的内容回答问题
- 明确引用来源文档和页码
- 对冲突信息标注"信息冲突：来源A认为...来源B认为..."

## 输出格式
**回答**：[基于检索内容的回答]
**来源**：[引用文档列表]
**检索建议**：[如果信息不足，建议补充的检索关键词]
```

### 2.4.2 提示词优化技巧
1. **指定回答风格**：
```markdown
# 回答风格
- 使用简洁的要点形式回答
- 每点不超过20个字
- 重点内容加粗显示
```

2. **多轮检索提示**：
```markdown
# 多轮检索优化
如果第一次检索结果不足以回答问题：
1. 分析缺失的信息
2. 生成补充检索关键词
3. 进行二次检索
4. 综合两次结果回答
```

## 2.5 课程作业：企业知识库系统开发

### 2.5.1 作业目标
使用Cherry Studio构建一个完整的企业知识库系统，实现文档管理、智能检索和问答功能。

### 2.5.2 具体要求
1. **知识库构建**：
   - 导入至少10篇不同类型的企业文档
   - 配置混合检索策略
   - 优化分块和嵌入参数

2. **应用开发**：
   - 创建知识库问答应用
   - 实现自定义提示词模板
   - 添加结果可视化功能

3. **性能优化**：
   - 对比不同检索策略的效果
   - 分析并优化低相关性查询案例
   - 生成性能测试报告

### 提交内容
- 知识库配置截图
- 应用界面截图
- 5组测试查询的问答记录
- 优化前后的性能对比报告
- 技术总结（500字以内）