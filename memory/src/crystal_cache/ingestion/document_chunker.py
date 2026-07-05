"""Document chunker — splits documents into content chunks by type.

This is Layer 1 of the dual-extraction strategy: content chunks for
traditional RAG retrieval. No LLM needed — pure text splitting.

Each chunk becomes a crystal with pair_type="content_chunk" so users
can ask "what's in Scene 5?" and get the actual text back.

Document type detection runs first, then type-specific chunking logic
splits the text into labeled chunks.

UNIFIED-KEY UPDATE: Each chunk includes a `locator` field used to build
the unified sparse key — a wide->specific '|' path (see
docs/UNIFIED_SPARSE_KEY.md). The caller provides the wider segments
(domain, subject, source); the chunker provides the locator, the most
specific segment.
"""
from __future__ import annotations

import ast
import re
from typing import Any

# File extensions that mark a document as source code. The extension on
# the upload label is the primary, most reliable code signal.
CODE_EXTENSIONS = (
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java",
    ".rb", ".c", ".h", ".cpp", ".hpp", ".cc", ".cs", ".php", ".swift",
    ".kt", ".scala", ".sh", ".bash",
)


def _looks_like_python(text: str) -> bool:
    """Heuristic: does this text look like Python source?

    Secondary signal when there is no file extension (e.g. code pasted into
    the upload box). Conservative: needs two of def / class / import.
    """
    head = text[:4000]
    sig = 0
    if re.search(r'^\s*def\s+\w+\s*\(', head, re.MULTILINE):
        sig += 1
    if re.search(r'^\s*class\s+\w+', head, re.MULTILINE):
        sig += 1
    if re.search(r'^\s*(from\s+\S+\s+)?import\s+\S', head, re.MULTILINE):
        sig += 1
    return sig >= 2


def detect_document_type(text: str, label: str = "") -> str:
    """Auto-detect document type from content and label.

    Returns one of: code, script, policy, contract, transcript, technical, general
    """
    text_lower = text[:5000].lower()
    label_lower = label.lower()

    # Code detection (highest priority): a source-file extension on the
    # label is the strongest signal. Falls back to a conservative content
    # check for code pasted without a filename.
    if label_lower.strip().endswith(CODE_EXTENSIONS):
        return "code"
    code_head = text[:8000]
    if "```" not in code_head:
        decls = len(re.findall(
            r'^(?:def |class |func |function |public |private )',
            code_head, re.MULTILINE,
        ))
        if decls >= 3 and _looks_like_python(text):
            return "code"

    # Script detection
    script_markers = [
        r'\bINT\.\s', r'\bEXT\.\s', r'\bFADE IN\b', r'\bFADE OUT\b',
        r'\bCUT TO\b', r'\(V\.O\.\)', r'\(O\.S\.\)', r"\(CONT'D\)",
    ]
    script_score = sum(1 for m in script_markers if re.search(m, text[:10000]))
    if script_score >= 2:
        return "script"

    caps_dialogue = re.findall(r'\n([A-Z]{2,}[A-Z\s]*)\n[^A-Z\n]', text[:10000])
    if len(caps_dialogue) >= 5:
        return "script"

    # Policy detection
    policy_markers = [
        r'\bpolicy\b', r'\bprocedure\b', r'\bstandard\b',
        r'\beffective date\b', r'\bapproved by\b', r'\brevision\b',
        r'\bcompliance\b', r'\bregulation\b',
    ]
    policy_score = sum(1 for m in policy_markers if re.search(m, text_lower))
    if policy_score >= 3 or any(w in label_lower for w in ['policy', 'procedure', 'standard', 'guideline']):
        return "policy"

    # Contract detection
    contract_markers = [
        r'\bwhereas\b', r'\bnow therefore\b', r'\bin witness whereof\b',
        r'\bherein\b', r'\bhereinafter\b', r'\bparty\b.*\bparty\b',
        r'\bagreement\b',
    ]
    contract_score = sum(1 for m in contract_markers if re.search(m, text_lower))
    if contract_score >= 3 or 'contract' in label_lower or 'agreement' in label_lower:
        return "contract"

    # Transcript detection
    transcript_patterns = [
        r'\b\d{1,2}:\d{2}\b', r'^[A-Z][a-z]+\s*:', r'\[.*?\]:',
    ]
    transcript_score = sum(1 for m in transcript_patterns if re.search(m, text[:5000], re.MULTILINE))
    if transcript_score >= 2 or any(w in label_lower for w in ['transcript', 'meeting', 'minutes', 'call']):
        return "transcript"

    # Technical detection
    tech_markers = [
        r'```', r'\bimport\s', r'\bdef\s+\w+\(', r'\bclass\s+\w+',
        r'https?://\S+/api/',
    ]
    tech_score = sum(1 for m in tech_markers if re.search(m, text[:5000]))
    if tech_score >= 2 or any(w in label_lower for w in ['readme', 'documentation', 'docs', 'api']):
        return "technical"

    return "general"


def chunk_document(text: str, doc_type: str, label: str = "") -> list[dict[str, Any]]:
    """Split document into labeled chunks based on type.

    Each chunk dict contains:
      - label: human-readable display label
      - text: the verbatim content
      - locator: the Locator field for the Elite Sparse Key
      - subject: (code only) the symbol name for the sparse key

    `label` carries the upload label (e.g. the filename); the code chunker
    uses it as the path component of each symbol's locator.
    """
    if doc_type == "code":
        return _chunk_code(text, label)
    elif doc_type == "script":
        return _chunk_script(text)
    elif doc_type == "policy":
        return _chunk_sections(text)
    elif doc_type == "contract":
        return _chunk_sections(text)
    elif doc_type == "transcript":
        return _chunk_transcript(text)
    elif doc_type == "technical":
        return _chunk_sections(text)
    else:
        return _chunk_general(text)


def _chunk_script(text: str) -> list[dict[str, Any]]:
    """Chunk a screenplay/script by scene breaks."""
    scene_pattern = re.compile(
        r'^\s*\d*\s*(?:INT\.|EXT\.|INT\./EXT\.|I/E\.)\s+.+$',
        re.MULTILINE | re.IGNORECASE
    )

    matches = list(scene_pattern.finditer(text))

    if not matches:
        return _chunk_general(text)

    chunks = []

    # Preamble before first scene
    if matches[0].start() > 50:
        preamble = text[:matches[0].start()].strip()
        if preamble:
            chunks.append({
                "label": "Title Page / Preamble",
                "text": preamble,
                "locator": "Title Page",
            })

    for i, match in enumerate(matches):
        header_line = match.group(0).strip()
        header_clean = re.sub(r'^\s*\d+\s+', '', header_line)
        header_clean = re.sub(r'\s+\d+\s*$', '', header_clean)

        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        scene_text = text[start:end].strip()

        if not scene_text:
            continue

        scene_num = i + 1
        label = f"Scene {scene_num}: {header_clean}"
        locator = f"Scene {scene_num}"

        chunks.append({
            "label": label,
            "text": scene_text,
            "locator": locator,
        })

    return chunks if chunks else _chunk_general(text)


def _chunk_sections(text: str) -> list[dict[str, Any]]:
    """Chunk by section headers (numbered or markdown-style)."""
    section_pattern = re.compile(
        r'^(?:'
        r'(?:\d+\.[\d.]*\s+\S)'
        r'|(?:Section\s+\d+)'
        r'|(?:Article\s+[IVXLCDM\d]+)'
        r'|(?:#{1,3}\s+\S)'
        r'|(?:[A-Z][A-Z\s]{3,}$)'
        r')',
        re.MULTILINE
    )

    matches = list(section_pattern.finditer(text))

    if len(matches) < 2:
        return _chunk_general(text)

    chunks = []

    if matches[0].start() > 100:
        preamble = text[:matches[0].start()].strip()
        if preamble:
            chunks.append({"label": "Introduction", "text": preamble, "locator": "Introduction"})

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_text = text[start:end].strip()
        if not section_text or len(section_text) < 20:
            continue

        first_line = section_text.split('\n')[0].strip()
        first_line = re.sub(r'^#+\s*', '', first_line)
        label = first_line[:100]
        locator = label  # Section header IS the locator

        chunks.append({"label": label, "text": section_text, "locator": locator})

    return chunks if chunks else _chunk_general(text)


def _chunk_transcript(text: str) -> list[dict[str, Any]]:
    """Chunk a transcript by speaker turns or time windows."""
    speaker_pattern = re.compile(r'^([A-Z][a-zA-Z\s]+?)\s*[:\-]\s*', re.MULTILINE)

    lines = text.split('\n')
    chunks = []
    current_chunk_lines: list[str] = []
    current_speakers: set[str] = set()
    char_count = 0

    for line in lines:
        m = speaker_pattern.match(line)
        if m:
            current_speakers.add(m.group(1).strip())

        current_chunk_lines.append(line)
        char_count += len(line)

        if char_count > 2000:
            chunk_text = '\n'.join(current_chunk_lines).strip()
            if chunk_text:
                speakers_str = ', '.join(sorted(current_speakers)[:3])
                label = f"Discussion ({speakers_str})" if speakers_str else f"Segment {len(chunks) + 1}"
                chunks.append({"label": label, "text": chunk_text, "locator": label})
            current_chunk_lines = []
            current_speakers = set()
            char_count = 0

    if current_chunk_lines:
        chunk_text = '\n'.join(current_chunk_lines).strip()
        if chunk_text:
            speakers_str = ', '.join(sorted(current_speakers)[:3])
            label = f"Discussion ({speakers_str})" if speakers_str else f"Segment {len(chunks) + 1}"
            chunks.append({"label": label, "text": chunk_text, "locator": label})

    return chunks if chunks else _chunk_general(text)


def _chunk_general(text: str, target_chars: int = 2000, overlap_chars: int = 200) -> list[dict[str, Any]]:
    """General-purpose chunking with sliding window."""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_text = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_text) + len(para) > target_chars and current_text:
            first_sentence = current_text.split('.')[0][:80]
            label = f"Passage {len(chunks) + 1}: {first_sentence}..."
            chunks.append({"label": label, "text": current_text, "locator": label})

            if overlap_chars > 0 and len(current_text) > overlap_chars:
                current_text = current_text[-overlap_chars:] + "\n\n" + para
            else:
                current_text = para
        else:
            current_text = (current_text + "\n\n" + para).strip() if current_text else para

    if current_text.strip():
        first_sentence = current_text.split('.')[0][:80]
        label = f"Passage {len(chunks) + 1}: {first_sentence}..."
        chunks.append({"label": label, "text": current_text, "locator": label})

    return chunks


def _node_source(lines: list[str], node: ast.AST) -> str:
    """Verbatim source for an AST node, including any decorators."""
    start = node.lineno
    dl = getattr(node, "decorator_list", None)
    if dl:
        start = min(start, min(d.lineno for d in dl))
    end = getattr(node, "end_lineno", node.lineno) or node.lineno
    return "".join(lines[start - 1:end]).rstrip("\n")


def _module_chunk(lines: list[str], path: str, start_line: int, end_line: int, idx: int):
    """Build a chunk for a run of module-level (non-symbol) code."""
    text = "".join(lines[start_line - 1:end_line]).strip()
    if not text:
        return None
    if idx == 0:
        locator, label = f"{path}::<module>", f"{path} (module)"
    else:
        locator, label = f"{path}::<module@L{start_line}>", f"{path} (module L{start_line})"
    return {"label": label, "text": text, "locator": locator,
            "subject": path.rsplit("/", 1)[-1]}


def _chunk_python(text: str, path: str) -> list[dict[str, Any]]:
    """Split Python source into one chunk per top-level symbol.

    Emits module-level code runs (imports, constants, __main__ block),
    each top-level function, a class-overview chunk per class, and one
    chunk per method. A symbol is never split mid-body. Each chunk's
    locator is `path::symbol` for identity retrieval.
    """
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    chunks: list[dict[str, Any]] = []
    mod_idx = 0
    buf_start = None
    prev_end = 0

    def flush(buf_end: int) -> None:
        nonlocal mod_idx, buf_start
        if buf_start is not None:
            mc = _module_chunk(lines, path, buf_start, buf_end, mod_idx)
            if mc:
                chunks.append(mc)
                mod_idx += 1
        buf_start = None

    for node in tree.body:
        is_sym = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        n_start = node.lineno
        dl = getattr(node, "decorator_list", None)
        if dl:
            n_start = min(n_start, min(d.lineno for d in dl))

        if not is_sym:
            if buf_start is None:
                buf_start = prev_end + 1 if prev_end else n_start
            prev_end = getattr(node, "end_lineno", node.lineno) or node.lineno
            continue

        flush(n_start - 1)

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append({"label": f"{node.name}()", "text": _node_source(lines, node),
                           "locator": f"{path}::{node.name}", "subject": node.name})
        else:
            methods = [m for m in node.body
                       if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            cls_start = n_start
            cls_end = methods[0].lineno - 1 if methods else (node.end_lineno or node.lineno)
            overview = "".join(lines[cls_start - 1:cls_end]).rstrip("\n")
            if overview.strip():
                chunks.append({"label": f"class {node.name}", "text": overview,
                               "locator": f"{path}::{node.name}", "subject": node.name})
            for m in methods:
                chunks.append({"label": f"{node.name}.{m.name}", "text": _node_source(lines, m),
                               "locator": f"{path}::{node.name}.{m.name}",
                               "subject": f"{node.name}.{m.name}"})
        prev_end = getattr(node, "end_lineno", node.lineno) or node.lineno

    if buf_start is not None:
        flush(len(lines))
    elif not chunks:
        mc = _module_chunk(lines, path, 1, len(lines), 0)
        if mc:
            chunks.append(mc)
    else:
        last = tree.body[-1]
        last_end = getattr(last, "end_lineno", last.lineno) or last.lineno
        if last_end < len(lines):
            mc = _module_chunk(lines, path, last_end + 1, len(lines), mod_idx)
            if mc:
                chunks.append(mc)
    return chunks


def _chunk_code_heuristic(text: str, path: str) -> list[dict[str, Any]]:
    """Best-effort per-symbol chunking for non-Python code.

    Splits on top-level declaration lines. Never crashes: with no
    declarations found, falls back to general chunking.
    """
    decl = re.compile(
        r'^(?:export\s+)?(?:public\s+|private\s+|protected\s+|static\s+|async\s+)*'
        r'(?:function|func|def|class|interface|type|struct|impl|fn)\s+([A-Za-z_]\w*)'
    )
    lines = text.splitlines(keepends=True)
    marks: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = decl.match(line)
        if m:
            marks.append((i, m.group(1)))
    if not marks:
        chunks = _chunk_general(text)
        for c in chunks:
            c["subject"] = path.rsplit("/", 1)[-1]
        return chunks
    chunks = []
    if marks[0][0] > 0:
        pre = "".join(lines[:marks[0][0]]).strip()
        if pre:
            chunks.append({"label": f"{path} (top)", "text": pre,
                           "locator": f"{path}::<top>",
                           "subject": path.rsplit("/", 1)[-1]})
    for idx, (start, sym) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else len(lines)
        body = "".join(lines[start:end]).rstrip("\n")
        if body.strip():
            chunks.append({"label": sym, "text": body,
                           "locator": f"{path}::{sym}", "subject": sym})
    return chunks


def _chunk_code(text: str, label: str = "") -> list[dict[str, Any]]:
    """Chunk source code by symbol. Python uses the ast module; other
    languages use a heuristic; unparseable input falls back to general
    chunking. Never raises on bad input.
    """
    path = (label or "code").strip()
    low = path.lower()
    if low.endswith(".py") or low.endswith(".pyi") or _looks_like_python(text):
        try:
            chunks = _chunk_python(text, path)
            if chunks:
                return chunks
        except SyntaxError:
            pass
    return _chunk_code_heuristic(text, path)
