from flask import request, render_template
from config import Config
from app import app
from pathlib import Path
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core import VectorStoreIndex, SummaryIndex
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.llms.openai import OpenAI
from typing import Optional, List
from llama_index.core.storage import StorageContext
import re

# Import your local Python functions
from local import get_ppv_stats, get_bridging

# Create FunctionTool instances for the Python functions
ppv_tool = FunctionTool.from_defaults(fn=get_ppv_stats)
bridging_tool = FunctionTool.from_defaults(fn=get_bridging)

def preprocess_query(query):
    """
    Convert percentages in the query to decimals.
    
    Example: "90%" -> 0.9
    """
    return re.sub(r"(\d+)%", lambda x: str(float(x.group(1)) / 100), query)

# Helper function to load the index and create tools
def load_index_and_create_tools(index_path, name):
    storage_dir = Path(index_path)
    docstore_path = storage_dir / "docstore.json"
    vectorstore_path = storage_dir / "vectorstore.json"

    if not docstore_path.exists() or not vectorstore_path.exists():
        raise FileNotFoundError(f"Required files not found in {storage_dir}")

    storage_context = StorageContext.from_defaults(
        docstore=docstore_path,
        vector_store=vectorstore_path
    )

    vector_index = VectorStoreIndex.from_storage_context(storage_context)

    def vector_query(query: str, page_numbers: Optional[List[str]] = None) -> str:
        page_numbers = page_numbers or []
        metadata_dict = [{"key": 'page_label', "value": p} for p in page_numbers]

        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(metadata_dict, condition=FilterCondition.OR)
        )

        response = query_engine.query(query)
        return response

    vector_query_tool = FunctionTool.from_defaults(
        name=f"vector_tool_{name}",
        fn=vector_query
    )

    summary_index = SummaryIndex(vector_index.nodes)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        se_async=True
    )

    summary_query_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name}",
        query_engine=summary_query_engine,
        description=("Use ONLY IF you want to get a holistic summary of the documents."
                     " DO NOT USE if you have specified questions over the documents.")
    )

    return vector_query_tool, summary_query_tool

@app.route("/", methods=["GET", "POST"])
def query_form():
    if request.method == "POST":
        query_text = request.form.get("query")
        if not query_text:
            return "Please enter a query.", 400

        try:
            # Preprocess the query to convert percentages to decimals
            query_text = preprocess_query(query_text)

            # Load and create tools for indexing
            output_directory = "./storage"
            tools = []
            
            for index_path in Path(output_directory).glob("*_index"):
                vector_tool, summary_tool = load_index_and_create_tools(index_path, index_path.stem)
                tools.extend([vector_tool, summary_tool])

            # Add custom tools for PPV and Bridging calculations
            tools.extend([ppv_tool, bridging_tool])

            # Create the agent worker
            OPENAI_API_KEY = Config.OPENAI_API_KEY
            llm = OpenAI(model="gpt-4", api_key=OPENAI_API_KEY)
            agent_worker = FunctionCallingAgentWorker.from_tools(tools, llm=llm, verbose=True)
            agent = AgentRunner(agent_worker=agent_worker)

            # Process the query using LLM
            response = agent.query(query_text)
            
            return render_template("index.html", response=response, query=query_text)
        except Exception as e:
            return str(e), 500

    return render_template("index.html", response=None)
