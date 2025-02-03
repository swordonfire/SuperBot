# scripts/clean_vector_db.py
import shutil
from pathlib import Path


def clean_chroma():
    chroma_path = Path('data/vector_db/chroma').absolute()
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
        print(f'Cleaned ChromaDB at {chroma_path}')
    else:
        print('No ChromaDB directory found')


if __name__ == '__main__':
    clean_chroma()
