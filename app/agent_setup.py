from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage import StorageContext
import os

import os

def create_and_save_indexes(directory, output_dir):
    papers = list(Path(directory).rglob("*.pdf"))
    
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    for paper in papers:
        print(f"Processing paper: {paper}")
        # Load and process the document
        documents = SimpleDirectoryReader(input_files=[paper]).load_data()
        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
        nodes = splitter.get_nodes_from_documents(documents)
        print(f"Length of nodes: {len(nodes)}")

        # Create the index
        vector_index = VectorStoreIndex(nodes)

        # Save the vector store and related data
        index_path = os.path.join(output_dir, f"{paper.stem}_index.json")

        # Persist the vector store to the specific file path
        vector_index.storage_context.vector_store.persist(persist_path=index_path)
        print(f"Index saved at: {index_path}")

