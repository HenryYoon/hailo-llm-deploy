"""Step 2: Extract court case numbers and statute references from answer text."""

import json
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# --- Compiled Regex Patterns (module-level constants) ---

CASE_PATTERN = re.compile(
    r'(대법원|헌법재판소|[가-힣]+(?:고등|지방)법원)'
    r'\s*'
    r'(\d{4})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*\.\s*'
    r'선고\s*'
    r'(\d{2,4}[가-힣]{1,2}\d+)'
    r'\s*'
    r'(판결|결정|전원합의체\s*(?:판결|결정))?'
)

STATUTE_BRACKET_PATTERN = re.compile(
    r'「([^」]+)」\s*(제\d+조(?:의\d+)?(?:\s*제\d+항)?(?:\s*제\d+호)?)'
)

STATUTE_PLAIN_PATTERN = re.compile(
    r'((?:[가-힣]{1,10}\s){0,1}[가-힣]{1,10}(?:법|령|규칙|규정|조례))'
    r'\s*(제\d+조(?:의\d+)?(?:\s*제\d+항)?(?:\s*제\d+호)?)'
)

RELATIVE_STATUTE_PATTERN = re.compile(
    r'(?:같은\s*법|동법)(\s*시행령|\s*시행규칙)?\s*'
    r'(제\d+조(?:의\d+)?(?:\s*제\d+항)?(?:\s*제\d+호)?)'
)

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

_CONTEXT_WORDS = {
    '결국', '경우', '그런데', '그렇지만', '다만', '달리', '또한', '모두',
    '무관하게', '보아', '아니라', '아닌', '않아', '역시', '여전히',
    '우리', '이상', '이후', '있어', '잡아', '지녀', '취득', '한편',
    '현행', '가령', '동법', '시행령', '시행규칙',
}

_PLAIN_FILTER_WORDS = {
    '같은', '위', '이', '그', '본', '해당', '방법', '처벌',
    '위반', '관련', '의한', '따른', '대한', '위한',
    '시행령', '시행규칙', '운용요령', '동법',
    '개정법', '법률', '보장법', '특례법',
}


class ReferenceExtractor:
    """Extract Korean legal citations (court cases and statute references) from text."""

    def __init__(
        self,
        sampled_paths: dict[str, Path],
        reference_extraction: Path,
        metadata_dir: Path,
    ):
        self.sampled_paths = sampled_paths
        self.reference_extraction = reference_extraction
        self.metadata_dir = metadata_dir

    @staticmethod
    def _trim_law_name(name: str) -> str:
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
        if result.startswith('구 '):
            result = result[2:]
        return result

    def extract_cases(self, text: str) -> list[dict]:
        """Extract court case references from answer text."""
        cases = []
        for match in CASE_PATTERN.finditer(text):
            cases.append({
                'court': match.group(1),
                'date': f"{match.group(2)}-{match.group(3).zfill(2)}-{match.group(4).zfill(2)}",
                'case_number': match.group(5),
                'judgment_type': (match.group(6) or "판결").strip(),
                'full_ref': match.group(0).strip(),
            })
        return cases

    def extract_statutes(self, text: str) -> list[dict]:
        """Extract statute references, resolving relative references."""
        statutes = []
        last_law_name = None
        all_matches = []

        for match in STATUTE_BRACKET_PATTERN.finditer(text):
            all_matches.append((match.start(), 'bracket', match))
        for match in RELATIVE_STATUTE_PATTERN.finditer(text):
            all_matches.append((match.start(), 'relative', match))
        for match in STATUTE_PLAIN_PATTERN.finditer(text):
            overlaps = any(
                mtype in ('bracket', 'relative')
                and match.start() >= m.start() and match.start() < m.end()
                for _, mtype, m in all_matches
            )
            if not overlaps:
                all_matches.append((match.start(), 'plain', match))

        all_matches.sort(key=lambda x: x[0])

        for _, mtype, match in all_matches:
            if mtype == 'bracket':
                law_name = match.group(1).strip()
                article = match.group(2).strip()
                last_law_name = law_name
            elif mtype == 'plain':
                law_name = self._trim_law_name(match.group(1).strip())
                article = match.group(2).strip()
                if law_name in _PLAIN_FILTER_WORDS:
                    continue
                if law_name.startswith('같은') or law_name.startswith('법률'):
                    continue
                if len(law_name) > 25 or len(law_name) < 2:
                    continue
                last_law_name = law_name
            elif mtype == 'relative':
                if last_law_name is None:
                    continue
                suffix = (match.group(1) or "").strip()
                article = match.group(2).strip()
                base_name = re.sub(r'\s*시행(?:령|규칙)$', '', last_law_name)
                law_name = base_name + (" " + suffix if suffix else "")

            statutes.append({
                'law_name': law_name,
                'article': article,
                'full_ref': f"{law_name} {article}",
            })

        seen = set()
        unique = []
        for s in statutes:
            if s['full_ref'] not in seen:
                seen.add(s['full_ref'])
                unique.append(s)
        return unique

    def extract_references(self, answer_text: str) -> dict:
        """Extract all legal references from a single answer text."""
        return {
            'cases': self.extract_cases(answer_text),
            'statutes': self.extract_statutes(answer_text),
        }

    def process_split(self, df: pd.DataFrame, split_name: str) -> list[dict]:
        """Process all rows in a split and return extraction results."""
        results = []
        for idx, row in df.iterrows():
            refs = self.extract_references(row['answer'])
            results.append({
                'index': idx,
                'split': split_name,
                'major_label': row['major_label'],
                'question': row['question'][:100],
                'cases': refs['cases'],
                'statutes': refs['statutes'],
                'num_cases': len(refs['cases']),
                'num_statutes': len(refs['statutes']),
            })
        return results

    def run(self) -> dict:
        """Execute reference extraction across all splits."""
        all_results = []
        for split_name, path in self.sampled_paths.items():
            df = pd.read_csv(path, dtype=str)
            results = self.process_split(df, split_name)
            all_results.extend(results)
            total_cases = sum(r['num_cases'] for r in results)
            total_statutes = sum(r['num_statutes'] for r in results)
            has_refs = sum(1 for r in results if r['num_cases'] > 0 or r['num_statutes'] > 0)
            logger.info("%s: %d samples, %d cases, %d statutes, %d with refs",
                        split_name, len(results), total_cases, total_statutes, has_refs)

        unique_cases = set()
        unique_law_names = set()
        for r in all_results:
            for c in r['cases']:
                unique_cases.add(c['case_number'])
            for s in r['statutes']:
                unique_law_names.add(s['law_name'])

        output = {
            'results': all_results,
            'summary': {
                'unique_case_numbers': sorted(unique_cases),
                'unique_statute_refs': sorted({s['full_ref'] for r in all_results for s in r['statutes']}),
                'unique_law_names': sorted(unique_law_names),
                'total_unique_cases': len(unique_cases),
                'total_unique_statutes': len({s['full_ref'] for r in all_results for s in r['statutes']}),
                'total_unique_law_names': len(unique_law_names),
            },
        }

        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        with open(self.reference_extraction, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info("Saved to %s", self.reference_extraction)
        return output
