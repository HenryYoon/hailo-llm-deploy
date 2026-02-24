"""
Step 5: Assemble RAFT dataset — oracle/distractor document selection + answer reformatting.
Produces final train/val/test JSON files compatible with datasets.load_dataset.
"""
import json
import re
import random

import pandas as pd

from config import (
    SAMPLED_TRAIN, SAMPLED_VAL, SAMPLED_TEST,
    REFERENCE_EXTRACTION, CHUNK_DATABASE,
    TRAIN_DATASET, VAL_DATASET, TEST_DATASET,
    PROCESSED_DATA, DOCS_PER_SAMPLE, MAX_ORACLE_DOCS,
    DISTRACTOR_SAME_LABEL_RATIO, RANDOM_SEED,
)

INSTRUCTION_TEMPLATE = (
    "당신은 한국의 법률 전문가입니다. 아래 제공된 참고 문서를 바탕으로 질문에 답변해주세요.\n"
    "답변에 참고한 문서는 [D1], [D2] 등으로 표시하고, 마지막에 참고문헌을 정리해주세요."
)


def _find_oracle_chunks(sample_refs, chunk_db):
    """Find oracle chunk IDs matching a sample's extracted references."""
    oracle_chunks = []

    for statute in sample_refs.get('statutes', []):
        law_name = statute['law_name']
        article = statute['article']
        # Search for matching chunks
        for chunk_id, chunk in chunk_db.items():
            if chunk['source_type'] in ('statute', 'statute_fallback'):
                if chunk['source_name'] == law_name and article in chunk.get('article', ''):
                    oracle_chunks.append(chunk_id)
                    break
        else:
            # Broader search: just law name match
            for chunk_id, chunk in chunk_db.items():
                if chunk['source_type'] in ('statute', 'statute_fallback'):
                    if chunk['source_name'] == law_name:
                        oracle_chunks.append(chunk_id)
                        break

    for case in sample_refs.get('cases', []):
        case_number = case['case_number']
        for chunk_id, chunk in chunk_db.items():
            if chunk['source_type'] in ('case', 'case_fallback'):
                if chunk['source_name'] == case_number:
                    oracle_chunks.append(chunk_id)
                    break

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for cid in oracle_chunks:
        if cid not in seen:
            seen.add(cid)
            unique.append(cid)
    return unique[:MAX_ORACLE_DOCS]


def _select_distractors(oracle_chunk_ids, major_label, chunk_db, label_to_chunks, num_needed):
    """Select distractor chunks: mix of same and different major_label."""
    oracle_set = set(oracle_chunk_ids)
    same_label_count = round(num_needed * DISTRACTOR_SAME_LABEL_RATIO)
    diff_label_count = num_needed - same_label_count

    distractors = []

    # Same major_label distractors
    same_pool = [cid for cid in label_to_chunks.get(major_label, []) if cid not in oracle_set]
    if same_pool:
        chosen = random.sample(same_pool, min(same_label_count, len(same_pool)))
        distractors.extend(chosen)

    # Different major_label distractors
    diff_pool = []
    for label, cids in label_to_chunks.items():
        if label != major_label:
            diff_pool.extend([cid for cid in cids if cid not in oracle_set])
    if diff_pool:
        remaining = num_needed - len(distractors)
        chosen = random.sample(diff_pool, min(remaining, len(diff_pool)))
        distractors.extend(chosen)

    # If still not enough, fill from any available chunk
    if len(distractors) < num_needed:
        all_pool = [cid for cid in chunk_db.keys() if cid not in oracle_set and cid not in distractors]
        remaining = num_needed - len(distractors)
        if all_pool:
            distractors.extend(random.sample(all_pool, min(remaining, len(all_pool))))

    return distractors[:num_needed]


def _build_label_to_chunks(chunk_db, ref_data):
    """Map major_label -> list of chunk_ids referenced by that label's samples."""
    label_to_chunks = {}

    for result in ref_data.get('results', []):
        label = result['major_label']
        if label not in label_to_chunks:
            label_to_chunks[label] = []

        for statute in result.get('statutes', []):
            law_name = statute['law_name']
            for chunk_id, chunk in chunk_db.items():
                if chunk['source_name'] == law_name:
                    label_to_chunks[label].append(chunk_id)

        for case in result.get('cases', []):
            case_number = case['case_number']
            for chunk_id, chunk in chunk_db.items():
                if chunk['source_name'] == case_number:
                    label_to_chunks[label].append(chunk_id)

    # Deduplicate per label
    for label in label_to_chunks:
        label_to_chunks[label] = list(set(label_to_chunks[label]))

    return label_to_chunks


def _format_doc_context(doc_indices_and_chunks):
    """Format documents as [D1] ..., [D2] ... text block."""
    lines = []
    for i, (_, chunk) in enumerate(doc_indices_and_chunks, 1):
        source = chunk['source_name']
        article = chunk.get('article', '')
        text = chunk['text']

        if chunk['source_type'] in ('statute', 'statute_fallback'):
            header = f"{source} {article}" if article else source
        else:
            court = chunk.get('court', '대법원')
            header = f"{court} {source} {article}"

        lines.append(f"[D{i}] {header}: {text}")
    return "\n".join(lines)


def _to_formal_style(text):
    """Convert plain-style (해라체) sentence endings to 습니다체 (formal polite)."""

    # Phase 1: Specific patterns (longest/most specific first)
    specific = [
        # Multi-word phrases
        (r'할 수 있다(?=\.)', '할 수 있습니다'),
        (r'할 수 없다(?=\.)', '할 수 없습니다'),
        (r'하여야 한다(?=\.)', '하여야 합니다'),
        (r'해야 한다(?=\.)', '해야 합니다'),
        (r'되어야 한다(?=\.)', '되어야 합니다'),
        (r'받아야 한다(?=\.)', '받아야 합니다'),
        (r'보아야 한다(?=\.)', '보아야 합니다'),
        (r'것이다(?=\.)', '것입니다'),
        (r'뿐이다(?=\.)', '뿐입니다'),
        (r'바이다(?=\.)', '바입니다'),
        # Copula / negation
        (r'아니다(?=\.)', '아닙니다'),
        (r'이다(?=\.)', '입니다'),
        # 하다/되다 (adjectives: 필요하다, 중요하다, 부당하다, etc.)
        (r'하다(?=\.)', '합니다'),
        (r'되다(?=\.)', '됩니다'),
        # ~ㄴ다 → ~ㅂ니다 (vowel-stem verb present tense)
        (r'한다(?=\.)', '합니다'),
        (r'된다(?=\.)', '됩니다'),
        (r'본다(?=\.)', '봅니다'),
        (r'준다(?=\.)', '줍니다'),
        (r'온다(?=\.)', '옵니다'),
        (r'간다(?=\.)', '갑니다'),
        (r'른다(?=\.)', '릅니다'),
        (r'든다(?=\.)', '듭니다'),
        # 르다 adjective/dictionary form (다르다, 모르다, etc.)
        (r'르다(?=\.)', '릅니다'),
    ]

    for pattern, repl in specific:
        text = re.sub(pattern, repl, text)

    # Phase 2: ~는다. → ~습니다. (consonant-stem verb present tense)
    text = re.sub(r'([\uAC00-\uD7A3])는다(?=\.)', r'\1습니다', text)

    # Phase 3: General ~다. handler (past tense, adjectives, etc.)
    def _convert_da(match):
        char = match.group(1)
        if char == '니':  # Part of already-converted ~ㅂ니다/습니다
            return match.group(0)
        code = ord(char)
        if 0xAC00 <= code <= 0xD7A3:
            final = (code - 0xAC00) % 28
            if final == 0:  # No 받침 → ambiguous (verb vs noun+copula), skip
                return match.group(0)
            elif final == 4:  # ㄴ 받침 → vowel-stem verb ㄴ다 → ㅂ니다
                return chr(code - 4 + 17) + '니다'
            else:  # Other consonant → 습니다
                return char + '습니다'
        return match.group(0)

    text = re.sub(r'([\uAC00-\uD7A3])다(?=\.)', _convert_da, text)

    return text


def _extract_quote(chunk_text, max_len=150):
    """Extract a concise quotable snippet from chunk text."""
    text = chunk_text.strip()
    # Find first meaningful sentence
    text = re.sub(r'\s+', ' ', text)
    if len(text) <= max_len:
        return text
    # Cut at sentence boundary
    cut = text[:max_len]
    m = list(re.finditer(r'[다요됨임함니까]\.|[.!?](?=\s|$)', cut))
    if m and m[-1].end() > 40:
        return cut[:m[-1].end()]
    return cut + "..."


def _build_cot_citations(oracle_doc_map, doc_pairs, oracle_set):
    """Build Chain-of-Thought citation block from oracle documents.

    Returns a string like:
        [D2]에 따르면, "민법 제107조: 의사표시는 표의자가 진의아님을 알고..."
        [D5]에 따르면, "대법원 2009다16766: 우리나라 영토 내에서..."
    """
    citations = []
    for di, (cid, chunk) in enumerate(doc_pairs, 1):
        if cid not in oracle_set:
            continue
        quote = _extract_quote(chunk['text'])
        source = chunk['source_name']
        article = chunk.get('article', '')
        if chunk['source_type'] in ('statute', 'statute_fallback'):
            header = f"{source} {article}".strip()
        else:
            court = chunk.get('court', '대법원')
            header = f"{court} {source}"
        citations.append(f"[D{di}]에 따르면, \"{header}: {quote}\"")
    return "\n".join(citations)


def _reformat_answer(original_answer, oracle_doc_map, sample_refs, doc_pairs, oracle_set):
    """
    Transform verbose answer to CoT citation format with [Dn] references.

    Format:
        [Dn]에 따르면, "문서 인용..."

        위 근거에 따르면, 답변 1~2문단

        참고문헌:
        - 조항 및 판례번호
    """
    answer = original_answer.strip()

    # 1. Remove long quoted blocks (various quote styles)
    answer = re.sub(r'「[^」]{80,}」', '', answer)
    answer = re.sub(r'"[^"]{80,}"', '', answer)
    answer = re.sub(r'\u201c[^\u201d]{80,}\u201d', '', answer)  # curly quotes

    # 2. Replace inline statute references with [Dn]
    for statute in sample_refs.get('statutes', []):
        full_ref = statute['full_ref']
        if full_ref in oracle_doc_map:
            dn = f"[D{oracle_doc_map[full_ref]}]"
            answer = answer.replace(full_ref, dn, 1)

    # 3. Replace inline case references with [Dn]
    for case in sample_refs.get('cases', []):
        full_ref = case['full_ref']
        ref_key = case['case_number']
        if ref_key in oracle_doc_map:
            dn = f"[D{oracle_doc_map[ref_key]}]"
            answer = answer.replace(full_ref, dn, 1)

    # 4. Compress to 1-2 paragraphs
    answer = re.sub(r'\s+', ' ', answer).strip()

    # Hard truncate at ~500 chars, then find last clean sentence ending
    if len(answer) > 500:
        cut = answer[:500]
        m = list(re.finditer(r'[다요됨임함니까]\.|[.!?](?=\s|$)', cut))
        if m and m[-1].end() > 100:
            answer = cut[:m[-1].end()]
        else:
            answer = cut

    # 5. Build CoT citation block from oracle documents
    cot_block = _build_cot_citations(oracle_doc_map, doc_pairs, oracle_set)

    # 6. Collect doc numbers used in the answer
    used_docs = sorted(set(re.findall(r'\[D\d+\]', answer)))
    if not used_docs and oracle_doc_map:
        doc_nums = sorted(set(oracle_doc_map.values()))
        used_docs = [f"[D{n}]" for n in doc_nums]
        answer += " (" + ", ".join(used_docs) + " 참조)"

    # 7. 참고문헌 section
    refs_section = "\n\n참고문헌:"
    for statute in sample_refs.get('statutes', []):
        refs_section += f"\n- {statute['full_ref']}"
    for case in sample_refs.get('cases', []):
        refs_section += f"\n- {case['court']} {case['case_number']} {case['judgment_type']}"

    # 8. Assemble: CoT citations → answer → references
    if cot_block:
        return f"{cot_block}\n\n위 근거에 따르면, {answer}{refs_section}"
    else:
        return f"{answer}{refs_section}"


def build_raft_dataset(df, split_name, ref_results, chunk_db, label_to_chunks):
    """Build RAFT format dataset for one split."""
    random.seed(RANDOM_SEED)

    # Index ref_results by (split, index) for quick lookup
    ref_lookup = {}
    for r in ref_results:
        if r['split'] == split_name:
            ref_lookup[r['index']] = r

    instructions = []
    inputs = []
    outputs = []
    skipped = 0

    for idx, row in df.iterrows():
        sample_refs = ref_lookup.get(idx, {})
        major_label = row['major_label']

        # Find oracle chunks
        oracle_chunk_ids = _find_oracle_chunks(sample_refs, chunk_db)

        # Select distractors
        num_distractors = DOCS_PER_SAMPLE - len(oracle_chunk_ids)
        if num_distractors < 0:
            oracle_chunk_ids = oracle_chunk_ids[:DOCS_PER_SAMPLE]
            num_distractors = 0

        distractor_ids = _select_distractors(
            oracle_chunk_ids, major_label, chunk_db, label_to_chunks, num_distractors
        )

        # Combine and shuffle
        all_doc_ids = oracle_chunk_ids + distractor_ids
        oracle_set = set(oracle_chunk_ids)

        # Pad if we don't have enough docs
        if len(all_doc_ids) < DOCS_PER_SAMPLE:
            remaining = DOCS_PER_SAMPLE - len(all_doc_ids)
            fill_pool = [cid for cid in chunk_db.keys() if cid not in set(all_doc_ids)]
            if fill_pool:
                all_doc_ids.extend(random.sample(fill_pool, min(remaining, len(fill_pool))))

        random.shuffle(all_doc_ids)

        # Build doc context and oracle map
        doc_pairs = []  # (chunk_id, chunk_data)
        for cid in all_doc_ids:
            if cid in chunk_db:
                doc_pairs.append((cid, chunk_db[cid]))

        if not doc_pairs:
            skipped += 1
            continue

        doc_context = _format_doc_context(doc_pairs)

        # Build oracle_doc_map: reference key -> Dn index
        oracle_doc_map = {}
        for di, (cid, chunk) in enumerate(doc_pairs, 1):
            if cid in oracle_set:
                # Map statute ref or case number to this Dn
                if 'statute' in chunk['source_type']:
                    key = f"{chunk['source_name']} {chunk.get('article', '')}".strip()
                    oracle_doc_map[key] = di
                else:
                    oracle_doc_map[chunk['source_name']] = di

        # Reformat answer with CoT citations
        reformatted = _reformat_answer(
            row['answer'], oracle_doc_map, sample_refs, doc_pairs, oracle_set,
        )

        # Build instruction with doc context
        instruction = f"{INSTRUCTION_TEMPLATE}\n\n{doc_context}"

        instructions.append(instruction)
        inputs.append(row['question'])
        outputs.append(reformatted)

    if skipped > 0:
        print(f"  [WARN] Skipped {skipped} samples (no chunks available)")

    return {
        "instruction": instructions,
        "input": inputs,
        "output": outputs,
    }


def run():
    """Execute Step 5: RAFT dataset assembly."""
    print("=" * 50)
    print("Step 5: RAFT Dataset Assembly")
    print("=" * 50)

    # Load inputs
    train_df = pd.read_csv(SAMPLED_TRAIN, dtype=str)
    val_df = pd.read_csv(SAMPLED_VAL, dtype=str)
    test_df = pd.read_csv(SAMPLED_TEST, dtype=str)

    with open(REFERENCE_EXTRACTION, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)

    with open(CHUNK_DATABASE, 'r', encoding='utf-8') as f:
        chunk_db = json.load(f)

    print(f"Loaded {len(chunk_db)} chunks")

    label_to_chunks = _build_label_to_chunks(chunk_db, ref_data)
    ref_results = ref_data['results']

    for df, split_name, output_path in [
        (train_df, 'train', TRAIN_DATASET),
        (val_df, 'val', VAL_DATASET),
        (test_df, 'test', TEST_DATASET),
    ]:
        print(f"\nBuilding {split_name} dataset ({len(df)} samples)...")
        dataset = build_raft_dataset(df, split_name, ref_results, chunk_db, label_to_chunks)
        print(f"  Output: {len(dataset['instruction'])} samples")

        PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for i in range(len(dataset['instruction'])):
                row = {
                    'instruction': dataset['instruction'][i],
                    'input': dataset['input'][i],
                    'output': dataset['output'][i],
                }
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f"  Saved: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    run()
