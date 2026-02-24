"""
Step 2: Extract court case numbers and statute references from answer text.
Uses regex patterns to parse Korean legal citations.
"""
import re
import json
import pandas as pd

from config import (
    SAMPLED_TRAIN, SAMPLED_VAL, SAMPLED_TEST,
    REFERENCE_EXTRACTION, METADATA,
)

# --- Regex Patterns ---

# Court case: 대법원 2007. 3. 30. 선고 2004다8333 판결
CASE_PATTERN = re.compile(
    r'(대법원|헌법재판소|[가-힣]+(?:고등|지방)법원)'
    r'\s*'
    r'(\d{4})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*\.\s*'
    r'선고\s*'
    r'(\d{2,4}[가-힣]{1,2}\d+)'
    r'\s*'
    r'(판결|결정|전원합의체\s*(?:판결|결정))?'
)

# Statute with 「」 brackets: 「근로기준법」 제34조
STATUTE_BRACKET_PATTERN = re.compile(
    r'「([^」]+)」'
    r'\s*'
    r'(제\d+조(?:의\d+)?(?:\s*제\d+항)?(?:\s*제\d+호)?)'
)

# Statute without brackets: 근로기준법 제34조, 민법 제750조
# At most 1 preceding word + law name word ending in 법/령/규칙/규정/조례
STATUTE_PLAIN_PATTERN = re.compile(
    r'((?:[가-힣]{1,10}\s){0,1}[가-힣]{1,10}(?:법|령|규칙|규정|조례))'
    r'\s*'
    r'(제\d+조(?:의\d+)?(?:\s*제\d+항)?(?:\s*제\d+호)?)'
)

# Sentence-context endings: particles, verb/adjective endings
_CONTEXT_ENDING_RE = re.compile(
    r'(?:'
    r'[을를이가은는에의도와과로서며고면나여해한된인란게히]'
    r'|으로|에서|에게|에는|에도|에서도|으로서|로서|으로써|로써|까지|부터|보다'
    r'|므로|이므로|하므로|으므로|였으므로'
    r'|하여|되어|하면|되면|하고|되고|하며|되며|하는|되는|있는|없는|있고|없고'
    r'|대해|대하여|대해서|대해서도|대하여는|관하여|관해'
    r'|위하여|위해서|위해|따라|따른|비추어|의하여'
    r'|라면|라고|이라|이란'
    r')$'
)

# Common sentence-context words that don't end with detectable particles
_CONTEXT_WORDS = {
    '결국', '경우', '그런데', '그렇지만', '다만', '달리', '또한', '모두',
    '무관하게', '보아', '아니라', '아닌', '않아', '역시', '여전히',
    '우리', '이상', '이후', '있어', '잡아', '지녀', '취득', '한편',
    '현행', '가령', '동법', '시행령', '시행규칙',
}


def _trim_law_name(name):
    """Remove leading sentence context words from a captured law name."""
    words = name.split()
    while len(words) > 1:
        first = words[0]
        if (len(first) == 1
                or _CONTEXT_ENDING_RE.search(first)
                or first in _CONTEXT_WORDS):
            words = words[1:]
        else:
            break
    result = ' '.join(words)
    # Strip "구 " prefix (old/former law) — API only has current names
    if result.startswith('구 '):
        result = result[2:]
    return result

# Relative reference: 같은 법 제5조, 같은 법 시행령 제3조, 동법 제5조
RELATIVE_STATUTE_PATTERN = re.compile(
    r'(?:같은\s*법|동법)'
    r'(\s*시행령|\s*시행규칙)?'
    r'\s*'
    r'(제\d+조(?:의\d+)?(?:\s*제\d+항)?(?:\s*제\d+호)?)'
)


def extract_cases(text):
    """Extract court case references from answer text."""
    cases = []
    for match in CASE_PATTERN.finditer(text):
        court = match.group(1)
        year = match.group(2)
        month = match.group(3)
        day = match.group(4)
        case_number = match.group(5)
        judgment_type = match.group(6) or "판결"

        cases.append({
            'court': court,
            'date': f"{year}-{month.zfill(2)}-{day.zfill(2)}",
            'case_number': case_number,
            'judgment_type': judgment_type.strip(),
            'full_ref': match.group(0).strip(),
        })
    return cases


def extract_statutes(text):
    """Extract statute references, resolving relative references."""
    statutes = []
    last_law_name = None

    # Combine bracket and plain patterns with their positions
    all_matches = []

    for match in STATUTE_BRACKET_PATTERN.finditer(text):
        all_matches.append((match.start(), 'bracket', match))

    for match in RELATIVE_STATUTE_PATTERN.finditer(text):
        all_matches.append((match.start(), 'relative', match))

    for match in STATUTE_PLAIN_PATTERN.finditer(text):
        # Skip if this overlaps with a bracket or relative match
        overlaps = False
        for _, mtype, m in all_matches:
            if mtype in ('bracket', 'relative') and (
                match.start() >= m.start() and match.start() < m.end()
            ):
                overlaps = True
                break
        if not overlaps:
            all_matches.append((match.start(), 'plain', match))

    # Sort by position in text
    all_matches.sort(key=lambda x: x[0])

    for pos, mtype, match in all_matches:
        if mtype == 'bracket':
            law_name = match.group(1).strip()
            article = match.group(2).strip()
            last_law_name = law_name
        elif mtype == 'plain':
            law_name = _trim_law_name(match.group(1).strip())
            article = match.group(2).strip()
            # Filter out false positives
            if law_name in ('같은', '위', '이', '그', '본', '해당', '방법', '처벌',
                            '위반', '관련', '의한', '따른', '대한', '위한',
                            '시행령', '시행규칙', '운용요령', '동법',
                            '개정법', '법률', '보장법', '특례법'):
                continue
            if law_name.startswith('같은') or law_name.startswith('법률'):
                continue
            if len(law_name) > 25:
                continue
            if len(law_name) < 2:
                continue
            last_law_name = law_name
        elif mtype == 'relative':
            if last_law_name is None:
                continue
            suffix = (match.group(1) or "").strip()
            article = match.group(2).strip()
            # Strip existing 시행령/시행규칙 from base name before appending
            base_name = re.sub(r'\s*시행(?:령|규칙)$', '', last_law_name)
            law_name = base_name + (" " + suffix if suffix else "")

        statutes.append({
            'law_name': law_name,
            'article': article,
            'full_ref': f"{law_name} {article}",
        })

    # Deduplicate by full_ref
    seen = set()
    unique_statutes = []
    for s in statutes:
        if s['full_ref'] not in seen:
            seen.add(s['full_ref'])
            unique_statutes.append(s)

    return unique_statutes


def extract_references(answer_text):
    """Extract all legal references from a single answer text."""
    return {
        'cases': extract_cases(answer_text),
        'statutes': extract_statutes(answer_text),
    }


def process_split(df, split_name):
    """Process all rows in a split and return extraction results."""
    results = []
    for idx, row in df.iterrows():
        refs = extract_references(row['answer'])
        results.append({
            'index': idx,
            'split': split_name,
            'major_label': row['major_label'],
            'question': row['question'][:100],  # truncated for metadata
            'cases': refs['cases'],
            'statutes': refs['statutes'],
            'num_cases': len(refs['cases']),
            'num_statutes': len(refs['statutes']),
        })
    return results


def run():
    """Execute Step 2: reference extraction."""
    print("=" * 50)
    print("Step 2: Legal Reference Extraction")
    print("=" * 50)

    train_df = pd.read_csv(SAMPLED_TRAIN, dtype=str)
    val_df = pd.read_csv(SAMPLED_VAL, dtype=str)
    test_df = pd.read_csv(SAMPLED_TEST, dtype=str)

    all_results = []
    for df, name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
        results = process_split(df, name)
        all_results.extend(results)
        total_cases = sum(r['num_cases'] for r in results)
        total_statutes = sum(r['num_statutes'] for r in results)
        has_refs = sum(1 for r in results if r['num_cases'] > 0 or r['num_statutes'] > 0)
        print(f"\n{name}: {len(results)} samples")
        print(f"  Cases found: {total_cases}")
        print(f"  Statutes found: {total_statutes}")
        print(f"  Samples with refs: {has_refs}/{len(results)} ({100*has_refs/len(results):.1f}%)")

    # Collect unique references across all splits
    unique_cases = set()
    unique_statutes = set()
    for r in all_results:
        for c in r['cases']:
            unique_cases.add(c['case_number'])
        for s in r['statutes']:
            unique_statutes.add(s['full_ref'])

    print(f"\nTotal unique case numbers: {len(unique_cases)}")
    print(f"Total unique statute refs: {len(unique_statutes)}")

    # Collect unique law names for API collection
    unique_law_names = set()
    for r in all_results:
        for s in r['statutes']:
            unique_law_names.add(s['law_name'])

    output = {
        'results': all_results,
        'summary': {
            'unique_case_numbers': sorted(unique_cases),
            'unique_statute_refs': sorted(unique_statutes),
            'unique_law_names': sorted(unique_law_names),
            'total_unique_cases': len(unique_cases),
            'total_unique_statutes': len(unique_statutes),
            'total_unique_law_names': len(unique_law_names),
        }
    }

    METADATA.mkdir(parents=True, exist_ok=True)
    with open(REFERENCE_EXTRACTION, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {REFERENCE_EXTRACTION}")

    return output


if __name__ == "__main__":
    run()
