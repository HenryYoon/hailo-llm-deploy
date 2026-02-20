"""
Step 4: Chunk collected statute and case texts into RAFT-appropriate segments.
Produces a chunk database mapping chunk_id -> chunk metadata & text.
Only chunks articles/cases that are actually referenced in the sample data.
"""
import json
import re

from config import (
    EXTERNAL_STATUTES, EXTERNAL_CASES, REFERENCE_EXTRACTION,
    CHUNK_DATABASE, METADATA, CHUNK_SIZE, CHUNK_OVERLAP,
)


def _extract_article_key(article_ref):
    """Extract the 조-level key from an article reference.

    e.g., '제246조 제1항 제8호' → '제246조'
          '제361조의5 제15호' → '제361조의5'
    """
    match = re.match(r'(제\d+조(?:의\d+)?)', article_ref)
    return match.group(1) if match else article_ref


def _format_article_text(article_data):
    """Format an article's data (content + paragraphs dict) into readable text."""
    content = article_data.get('content', '')
    paragraphs = article_data.get('paragraphs', {})

    parts = [content] if content else []

    for p_key in sorted(paragraphs.keys()):
        p_data = paragraphs[p_key]
        p_content = p_data.get('content', '')
        items = p_data.get('items', [])

        if p_content:
            parts.append(p_content)
        for item in items:
            parts.append(f"  {item}")

    return "\n".join(parts)


def _get_referenced_articles(ref_data):
    """Build mapping: law_name → set of article keys (조 level) from references."""
    law_articles = {}
    referenced_cases = set()

    for result in ref_data.get('results', []):
        for statute in result.get('statutes', []):
            law_name = statute['law_name']
            article_key = _extract_article_key(statute['article'])
            if law_name not in law_articles:
                law_articles[law_name] = set()
            law_articles[law_name].add(article_key)

        for case in result.get('cases', []):
            referenced_cases.add(case['case_number'])

    return law_articles, referenced_cases


def chunk_by_sentences(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks at sentence boundaries."""
    if not text or not text.strip():
        return []

    sentences = re.split(r'(?<=[.다요함임됨])\s+|\n+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk = (current_chunk + " " + sentence).strip()

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def chunk_statute(law_name, statute_data, referenced_article_keys):
    """Chunk a statute into article-level chunks for referenced articles only."""
    chunks = []

    if 'error' in statute_data:
        return chunks

    articles = statute_data.get('articles', {})
    for article_key, article_data in articles.items():
        if article_key not in referenced_article_keys:
            continue

        full_text = _format_article_text(article_data)
        if not full_text.strip():
            continue

        if len(full_text) <= CHUNK_SIZE:
            chunks.append({
                'chunk_id': f"statute_{law_name}_{article_key}",
                'source_type': 'statute',
                'source_name': law_name,
                'article': article_key,
                'text': full_text.strip(),
            })
        else:
            sub_chunks = chunk_by_sentences(full_text)
            for ci, sc in enumerate(sub_chunks):
                chunks.append({
                    'chunk_id': f"statute_{law_name}_{article_key}_c{ci}",
                    'source_type': 'statute',
                    'source_name': law_name,
                    'article': article_key,
                    'text': sc,
                })

    return chunks


def chunk_case(case_number, case_data):
    """Chunk a court case into holdings/summary chunks."""
    chunks = []

    if 'error' in case_data:
        return chunks

    sections = [
        ('summary', case_data.get('summary', '')),
        ('holdings', case_data.get('holdings', '')),
    ]

    for section_name, section_text in sections:
        if not section_text or not section_text.strip():
            continue

        # Clean HTML tags
        section_text = section_text.replace('<br/>', '\n').replace('<br>', '\n')
        section_text = re.sub(r'<[^>]+>', '', section_text)
        section_text = section_text.strip()

        sub_chunks = chunk_by_sentences(section_text)
        for ci, sc in enumerate(sub_chunks):
            chunks.append({
                'chunk_id': f"case_{case_number}_{section_name}_c{ci}",
                'source_type': 'case',
                'source_name': case_number,
                'article': section_name,
                'text': sc,
                'court': case_data.get('court', ''),
                'case_name': case_data.get('case_name', ''),
            })

    return chunks


def build_chunk_database(ref_data):
    """Build the chunk database from external data, only for referenced articles."""
    all_chunks = {}

    law_articles, referenced_cases = _get_referenced_articles(ref_data)
    print(f"Referenced: {len(law_articles)} law names, {len(referenced_cases)} case numbers")

    # Process statutes — only referenced law names and their specific articles
    statute_files = list(EXTERNAL_STATUTES.glob("*.json"))
    statutes_processed = 0
    articles_chunked = 0

    for sf in statute_files:
        with open(sf, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'error' in data:
            continue

        # Match by filename-derived name (= extracted name from reference_extraction)
        stem_name = sf.stem.replace('_', ' ')
        referenced_keys = law_articles.get(stem_name)

        if referenced_keys is None:
            continue

        chunks = chunk_statute(stem_name, data, referenced_keys)
        for chunk in chunks:
            all_chunks[chunk['chunk_id']] = chunk

        statutes_processed += 1
        articles_chunked += len(chunks)

    print(f"Statutes processed: {statutes_processed}, chunks: {articles_chunked}")

    # Process cases — only referenced case numbers
    cases_processed = 0
    case_chunks_count = 0

    for cf in EXTERNAL_CASES.glob("*.json"):
        case_number = cf.stem
        if case_number not in referenced_cases:
            continue

        with open(cf, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chunks = chunk_case(case_number, data)
        for chunk in chunks:
            all_chunks[chunk['chunk_id']] = chunk

        cases_processed += 1
        case_chunks_count += len(chunks)

    print(f"Cases processed: {cases_processed}, chunks: {case_chunks_count}")

    return all_chunks


def build_fallback_chunks(ref_data):
    """Build fallback chunks for references that couldn't be fetched."""
    fallback_chunks = {}

    # Load existing statute data indexed by filename stem
    existing_statutes = {}
    for sf in EXTERNAL_STATUTES.glob("*.json"):
        with open(sf, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'error' not in data:
            stem_name = sf.stem.replace('_', ' ')
            existing_statutes[stem_name] = data

    existing_cases = set()
    for cf in EXTERNAL_CASES.glob("*.json"):
        with open(cf, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'error' not in data:
            existing_cases.add(cf.stem)

    for result in ref_data.get('results', []):
        for statute in result.get('statutes', []):
            law_name = statute['law_name']
            article_key = _extract_article_key(statute['article'])

            if law_name in existing_statutes:
                articles = existing_statutes[law_name].get('articles', {})
                if article_key in articles:
                    continue

            full_ref = statute['full_ref']
            chunk_id = f"fallback_statute_{full_ref.replace(' ', '_')}"
            if chunk_id not in fallback_chunks:
                fallback_chunks[chunk_id] = {
                    'chunk_id': chunk_id,
                    'source_type': 'statute_fallback',
                    'source_name': law_name,
                    'article': statute['article'],
                    'text': f"{full_ref} (원문 미수집)",
                }

        for case in result.get('cases', []):
            if case['case_number'] not in existing_cases:
                chunk_id = f"fallback_case_{case['case_number']}"
                if chunk_id not in fallback_chunks:
                    fallback_chunks[chunk_id] = {
                        'chunk_id': chunk_id,
                        'source_type': 'case_fallback',
                        'source_name': case['case_number'],
                        'article': 'fallback',
                        'text': f"{case['full_ref']} (원문 미수집)",
                        'court': case.get('court', ''),
                    }

    return fallback_chunks


def run():
    """Execute Step 4: chunk text from collected data."""
    print("=" * 50)
    print("Step 4: Document Chunking")
    print("=" * 50)

    with open(REFERENCE_EXTRACTION, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)

    all_chunks = build_chunk_database(ref_data)
    print(f"\nExternal chunks created: {len(all_chunks)}")

    fallback_chunks = build_fallback_chunks(ref_data)
    print(f"Fallback chunks created: {len(fallback_chunks)}")

    all_chunks.update(fallback_chunks)
    print(f"Total chunks: {len(all_chunks)}")

    statute_chunks = sum(1 for c in all_chunks.values() if 'statute' in c['source_type'])
    case_chunks = sum(1 for c in all_chunks.values() if 'case' in c['source_type'])
    print(f"  Statute chunks: {statute_chunks}")
    print(f"  Case chunks: {case_chunks}")

    METADATA.mkdir(parents=True, exist_ok=True)
    with open(CHUNK_DATABASE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {CHUNK_DATABASE}")

    return all_chunks


if __name__ == "__main__":
    run()
