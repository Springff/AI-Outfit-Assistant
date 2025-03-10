# 设置OpenMP线程数为8
import os
os.environ["OMP_NUM_THREADS"] = "8"

from typing import Any, List, Optional

# 从llama_index库导入HuggingFaceEmbedding类，用于将文本转换为向量表示
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# 从llama_index库导入ChromaVectorStore类，用于高效存储和检索向量数据
from llama_index.vector_stores.chroma import ChromaVectorStore
# 从llama_index库导入PyMuPDFReader类，用于读取和解析PDF文件内容
from llama_index.readers.file import PyMuPDFReader
# 从llama_index库导入NodeWithScore和TextNode类
# NodeWithScore: 表示带有相关性分数的节点，用于排序检索结果
# TextNode: 表示文本块，是索引和检索的基本单位。节点存储文本内容及其元数据，便于构建知识图谱和语义搜索
from llama_index.core.schema import NodeWithScore, TextNode
# 从llama_index库导入RetrieverQueryEngine类，用于协调检索器和响应生成，执行端到端的问答过程
from llama_index.core.query_engine import RetrieverQueryEngine
# 从llama_index库导入QueryBundle类，用于封装查询相关的信息，如查询文本、过滤器等
from llama_index.core import QueryBundle
# 从llama_index库导入BaseRetriever类，这是所有检索器的基类，定义了检索接口
from llama_index.core.retrievers import BaseRetriever
# 从llama_index库导入SentenceSplitter类，用于将长文本分割成句子或语义完整的文本块，便于索引和检索
from llama_index.core.node_parser import SentenceSplitter
# 从llama_index库导入VectorStoreQuery类，用于构造向量存储的查询，支持语义相似度搜索
from llama_index.core.vector_stores import VectorStoreQuery
# 向量数据库
import chromadb
from ipex_llm.llamaindex.llms import IpexLLM
import time


class Config:
    """配置类,存储所有需要的参数"""
    model_path = "qwen2chat_int8"
    tokenizer_path = "qwen2chat_int8"
    question = "我今天该穿什么？"
    data_path = "./doc"
    persist_dir = "./chroma_db_test"
    embedding_model_path = "qwen2chat_src/AI-ModelScope/bge-small-zh-v1___5"
    max_new_tokens = 8192

def load_vector_database(persist_dir: str) -> ChromaVectorStore:
    # 检查持久化目录是否存在
    if os.path.exists(persist_dir):
        print(f"正在加载现有的向量数据库: {persist_dir}")
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = chroma_client.get_collection("dress")
    else:
        print(f"创建新的向量数据库: {persist_dir}")
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = chroma_client.create_collection("dress")
    print(f"Vector store loaded with {chroma_collection.count()} documents")
    return ChromaVectorStore(chroma_collection=chroma_collection)

def load_data(data_path: str) -> List[TextNode]:
    loader = PyMuPDFReader()
    documents = loader.load(file_path=data_path)

    text_parser = SentenceSplitter(chunk_size=384)
    text_chunks = []
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(text=text_chunk)
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)
    return nodes

class VectorDBRetriever(BaseRetriever):
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        print(f"Retrieved {len(nodes_with_scores)} nodes with scores")
        return nodes_with_scores

def completion_to_prompt(completion: str) -> str:
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"

def messages_to_prompt(messages: List[dict]) -> str:
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == "user":
            prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == "assistant":
            prompt += f"<|assistant|>\n{message.content}</s>\n"

    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt

    prompt = prompt + "<|assistant|>\n"

    return prompt

def setup_llm(config: Config) -> IpexLLM:
    """
    设置语言模型
    
    Args:
        config (Config): 配置对象
    
    Returns:
        IpexLLM: 配置好的语言模型
    """
    return IpexLLM.from_model_id_low_bit(
        model_name=config.model_path,
        tokenizer_name=config.tokenizer_path,
        context_window=16384,
        max_new_tokens=config.max_new_tokens,
        generate_kwargs={"temperature": 0.7, "do_sample": True},
        model_kwargs={},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        device_map="cpu",
    )

def generate_response(message):
    config = Config()
    embed_model = HuggingFaceEmbedding(model_name=config.embedding_model_path)
    llm = setup_llm(config)
    vector_store = load_vector_database(persist_dir=config.persist_dir)
    query_str = message
    query_embedding = embed_model.get_query_embedding(query_str)
    query_mode = "default"
    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
    )
    query_result = vector_store.query(vector_store_query)

    nodes_with_scores = []
    for index, node in enumerate(query_result.nodes):
        score: Optional[float] = None
        if query_result.similarities is not None:
            score = query_result.similarities[index]
        nodes_with_scores.append(NodeWithScore(node=node, score=score))
    
    # 设置检索器
    retriever = VectorDBRetriever(
        vector_store, embed_model, query_mode="default", similarity_top_k=10
    )
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
    response = query_engine.query(query_str)
    return response


def main():
    """主函数"""
    config = Config() 
    time_vec_db_start = time.time()
    # 数据库插入下载一次
    if os.path.exists(config.persist_dir) and os.listdir(config.persist_dir):
        print("数据库已加载")
    else:
        embed_model = HuggingFaceEmbedding(model_name=config.embedding_model_path)
        # 加载向量数据库
        vector_store = load_vector_database(persist_dir=config.persist_dir)
        for filename in os.listdir(config.data_path):
            file_path = os.path.join(config.data_path, filename)    
            nodes = load_data(data_path=file_path)
            for node in nodes:
                node_embedding = embed_model.get_text_embedding(
                    node.get_content(metadata_mode="all")
                )
                node.embedding = node_embedding
            vector_store.add(nodes)
    time_vec_db_end = time.time()
    time_vec_db_lat = time_vec_db_end - time_vec_db_start

    time_generate_response_start = time.time()
    init_info = f"用户的性别是女性，脸型是长脸，宽肩、长臂、H 型体型，身高是 170 cm，体重是 55 kg，目标场景在参加朋友的婚礼，颜色偏好为黄色，喜爱的风格偏好为简约、经典，另外你推荐的穿搭还有舒适性，能迅速抢到捧花的要求。现在你是一位虚拟的时尚顾问，请需要根据上面提供的这些信息给出具体的搭配建议，你需要根据用户提供的场景信息，推荐一套完整的服饰搭配，包括上衣、下装、鞋子和配件等，并简要说明每个搭配的原因，一条一条来。"
    
    answer = generate_response(init_info)
    time_generate_response_end = time.time()
    time_generate_response_lat = time_generate_response_end - time_generate_response_start
    print('vec_db:%.2f, gen_response:%.2f' %(time_vec_db_lat, time_generate_response_lat))




if __name__ == "__main__":
    main()