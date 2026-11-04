from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, DirectoryLoader, GitLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader



def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the diata directory and converst to LangChain
    Document Structure.

    Supported : PDF, TXT, CSV, Excel, Word, JSON, GitHub Repo
    """

    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents = []

    # PDF files

    pdf_files = list((data_path / "pdf_file").glob("*.pdf"))

    for pdf_file in pdf_files:
        try: 
            loader = PyMuPDFLoader(str(pdf_file))
            loaded = loader.load()
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] failed to load {pdf_file}: {e}")

    
    return documents

    