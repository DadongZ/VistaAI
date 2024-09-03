from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.tools import FunctionTool
from llama_index.core.vector_stores import FilterCondition
from typing import Optional, List

def get_router_query_engine(file_path: str, llm = None, embed_model = None):
    """Get router query engine."""
    llm = llm or OpenAI(model="gpt-3.5-turbo")
    embed_model = embed_model or OpenAIEmbedding(model="text-embedding-ada-002")
    
    # load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    
    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes, embed_model=embed_model)
    
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
        llm=llm
    )
    vector_query_engine = vector_index.as_query_engine(llm=llm)
    
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=(
            "Useful for summarization questions related to MetaGPT"
        ),
    )
    
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Useful for retrieving specific context from the MetaGPT paper."
        ),
    )
    
    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,
            vector_tool,
        ],
        verbose=True
    )
    return query_engine

from llama_index import SimpleDirectoryReader, SentenceSplitter, VectorStoreIndex
from llama_index.core.tools import FunctionTool, QueryEngineTool

def get_doc_tools(file_path: str, name: str):
    '''
    Get vector query and summary query tools from a document.
    '''
    # Load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    print("Length of nodes")
    
    # Split documents into nodes
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Length of nodes : {len(nodes)}")
    
    # Instantiate VectorStore without storage_context
    vector_index = VectorStoreIndex(nodes)  # Skip storage_context if it's optional

    # Persist the vector store if required
    vector_index.storage_context.vector_store.persist(persist_path="./data/chroma_db")
    
    # Calculate the maximum allowable length for 'name'
    max_name_length = 60 - len("vector_tool_")

    # Ensure truncation is handled correctly
    truncated_name = name[:max_name_length] if len(name) >= max_name_length else name

    print(f"Original name: {name}, Truncated name: {truncated_name}")

    # Define VectorStore Autoretrieval tool
    def vector_query(query: str, page_numbers: Optional[List[str]] = None) -> str:
        page_numbers = page_numbers or []
        metadata_dict = [{"key": 'page_label', "value": p} for p in page_numbers]

        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(metadata_dict, condition=FilterCondition.OR)
        )
        
        response = query_engine.query(query)
        return response
    
    # LlamaIndex FunctionTool wraps any Python function we feed it
    vector_query_tool = FunctionTool.from_defaults(
        name=f"vector_tool_{truncated_name}",
        fn=vector_query
    )

    # Prepare Summary Tool
    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        se_async=True
    )
    
    summary_query_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{truncated_name}",
        query_engine=summary_query_engine,
        description=("Use ONLY IF you want to get a holistic summary of the documents."
                     " DO NOT USE if you have specified questions over the documents.")
    )
    return vector_query_tool, summary_query_tool
