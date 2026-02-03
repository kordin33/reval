"""
Extract clean text from LaTeX file for AI detection testing.
Removes: LaTeX commands, code listings, math, figures, tables.
"""

import re
from pathlib import Path

def extract_text_from_latex(latex_path: str) -> str:
    """Extract readable text from LaTeX file."""
    
    with open(latex_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove everything before \begin{document}
    match = re.search(r'\\begin\{document\}', content)
    if match:
        content = content[match.end():]
    
    # Remove \end{document}
    content = re.sub(r'\\end\{document\}', '', content)
    
    # Remove lstlisting environments (code blocks)
    content = re.sub(r'\\begin\{lstlisting\}.*?\\end\{lstlisting\}', '[KOD ŹRÓDŁOWY POMINIĘTY]', content, flags=re.DOTALL)
    
    # Remove verbatim environments
    content = re.sub(r'\\begin\{verbatim\}.*?\\end\{verbatim\}', '[STRUKTURA KATALOGÓW]', content, flags=re.DOTALL)
    
    # Remove figure environments but keep captions
    content = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', content, flags=re.DOTALL)
    
    # Remove table contents but not completely
    content = re.sub(r'\\begin\{tabular\}.*?\\end\{tabular\}', '[TABELA]', content, flags=re.DOTALL)
    
    # Remove inline math
    content = re.sub(r'\$[^$]+\$', '', content)
    
    # Remove display math
    content = re.sub(r'\\\[.*?\\\]', '', content, flags=re.DOTALL)
    
    # Remove footnotes but extract text
    content = re.sub(r'\\footnote\{[^}]*\}', '', content)
    
    # Remove common LaTeX commands
    commands_to_remove = [
        r'\\chapter\{([^}]*)\}',
        r'\\section\{([^}]*)\}', 
        r'\\subsection\{([^}]*)\}',
        r'\\subsubsection\{([^}]*)\}',
        r'\\textbf\{([^}]*)\}',
        r'\\textit\{([^}]*)\}',
        r'\\emph\{([^}]*)\}',
        r'\\texttt\{([^}]*)\}',
        r'\\url\{[^}]*\}',
        r'\\label\{[^}]*\}',
        r'\\ref\{[^}]*\}',
        r'\\caption\{[^}]*\}',
    ]
    
    # Replace commands keeping their content
    for pattern in commands_to_remove:
        if '(' in pattern:  # has capture group
            content = re.sub(pattern, r'\1', content)
        else:
            content = re.sub(pattern, '', content)
    
    # Remove remaining LaTeX commands
    content = re.sub(r'\\[a-zA-Z]+\*?\{[^}]*\}', '', content)
    content = re.sub(r'\\[a-zA-Z]+\*?', '', content)
    
    # Remove special characters
    content = content.replace('\\&', '&')
    content = content.replace('\\%', '%')
    content = content.replace('\\$', '$')
    content = content.replace('~', ' ')
    content = content.replace('``', '"')
    content = content.replace("''", '"')
    
    # Remove itemize/enumerate markers
    content = re.sub(r'\\begin\{itemize\}', '', content)
    content = re.sub(r'\\end\{itemize\}', '', content)
    content = re.sub(r'\\begin\{enumerate\}', '', content)
    content = re.sub(r'\\end\{enumerate\}', '', content)
    content = re.sub(r'\\item\s*', '\n• ', content)
    
    # Clean up whitespace
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    content = re.sub(r'[ \t]+', ' ', content)
    
    # Remove empty lines with only spaces
    lines = content.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    
    return '\n\n'.join(lines)


def split_into_chunks(text: str, chunk_size: int = 2000) -> list:
    """Split text into chunks for AI detection (most have char limits)."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_len = 0
    
    for word in words:
        if current_len + len(word) + 1 > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_len = len(word)
        else:
            current_chunk.append(word)
            current_len += len(word) + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def extract_from_plain_text(text_path: str) -> str:
    """Wczytaj czysty tekst z pliku .txt / .md."""
    with open(text_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_text(file_path: str) -> str:
    """Auto-detect format i wyekstrahuj tekst."""
    p = Path(file_path)
    if p.suffix.lower() == '.tex':
        return extract_text_from_latex(file_path)
    else:
        return extract_from_plain_text(file_path)


def process_file(input_path: str, output_dir: str = None, chunk_size: int = 2500):
    """Wyekstrahuj tekst, podziel na chunki, zapisz."""
    out = Path(output_dir) if output_dir else Path(__file__).parent
    chunks_dir = out / "chunks"

    # Wyczysc stare chunki
    if chunks_dir.exists():
        for old in chunks_dir.glob("chunk_*.txt"):
            old.unlink()

    text = extract_text(input_path)

    # Zapisz pelny tekst
    full_text_path = out / "full_text.txt"
    full_text_path.write_text(text, encoding='utf-8')
    print(f"Pelny tekst: {full_text_path} ({len(text)} znakow)")

    # Podziel na chunki
    chunks = split_into_chunks(text, chunk_size=chunk_size)
    chunks_dir.mkdir(exist_ok=True)

    for i, chunk in enumerate(chunks):
        chunk_path = chunks_dir / f"chunk_{i+1:02d}.txt"
        chunk_path.write_text(chunk, encoding='utf-8')

    print(f"Podzielono na {len(chunks)} chunkow w: {chunks_dir}")
    return chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uzycie: python extract_text.py <plik.tex|plik.txt> [chunk_size]")
        print("  Przyklad: python extract_text.py praca.tex")
        print("  Przyklad: python extract_text.py praca.txt 3000")
        sys.exit(1)

    input_file = sys.argv[1]
    csize = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
    process_file(input_file, chunk_size=csize)
