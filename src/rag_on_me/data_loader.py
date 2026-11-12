from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from rag_on_me.loading_git import load_github_repos




def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert them to LangChain
    Documents.

    Supported: PDFs in pdf_file/, Markdown profile docs in profile/, GitHub repos
    listed in github/repos.txt
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
            for doc in loaded:                         # OS-independent name for citations
                doc.metadata["source_type"] = "pdf"
                doc.metadata["source_name"] = pdf_file.name
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] failed to load {pdf_file}: {e}")


    # Profile (Markdown) files

    for md_file in (data_path / "profile").glob("*.md"):
        try:
            loaded = TextLoader(str(md_file), encoding="utf-8").load()
            for doc in loaded:
                doc.metadata["source_type"] = "profile"
                doc.metadata["source_name"] = md_file.name
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] failed to load {md_file}: {e}")


    # GitHub files

    repo_list = data_path / "github" / "repos.txt"
    if repo_list.exists():
        documents.extend(load_github_repos(repo_list, data_path / "repo_cache"))



    return documents


