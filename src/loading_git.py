
from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, DirectoryLoader, GitLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader


def load_github_repos(repo_list_file: Path, clone_dir: Path) -> List[Any]:
    """Clone each repo listed in repo_list_file and load its README.md and requirements.txt."""
    documents: List[Any] = []
    wanted_files = {"README.md", "requirements.txt"}

    for line in repo_list_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):          # skip blanks and comments
            continue

        parts = line.split()
        url = parts[0]
        branch = parts[1] if len(parts) > 1 else "main"
        repo_name = url.rstrip("/").split("/")[-1].removesuffix(".git")

        try:
            loader = GitLoader(
                clone_url=url,
                repo_path=str(clone_dir / repo_name),
                branch=branch,
                file_filter=lambda p: Path(p).name in wanted_files,
            )
            loaded = loader.load()

            for doc in loaded:                         # tag so answers can cite the repo
                doc.metadata["source_type"] = "github"
                doc.metadata["repo"] = repo_name
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] failed to load repo {url}: {e}")
    return documents