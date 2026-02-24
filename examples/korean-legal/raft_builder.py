"""Step 5: Assemble RAFT dataset with oracle/distractor document selection."""

import json
import logging
import random
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class RaftBuilder:
    """Assemble RAFT dataset with oracle documents, distractors, and CoT citations.

    Encapsulates chunk_db and label_to_chunks to eliminate parameter threading.
    """

    def __init__(
        self,
        sampled_paths: dict[str, Path],
        reference_extraction: Path,
        chunk_database: Path,
        dataset_paths: dict[str, Path],
        processed_data: Path,
        docs_per_sample: int = 5,
        max_oracle_docs: int = 2,
        distractor_same_label_ratio: float = 0.6,
        random_seed: int = 42,
        instruction_template: str = (
            "당신은 한국의 법률 전문가입니다. 아래 제공된 참고 문서를 바탕으로 질문에 답변해주세요.\n"
            "답변에 참고한 문서는 [D1], [D2] 등으로 표시하고, 마지막에 참고문헌을 정리해주세요."
        ),
    ):
        self.sampled_paths = sampled_paths
        self.reference_extraction = reference_extraction
        self.chunk_database_path = chunk_database
        self.dataset_paths = dataset_paths
        self.processed_data = processed_data
        self.docs_per_sample = docs_per_sample
        self.max_oracle_docs = max_oracle_docs
        self.distractor_same_label_ratio = distractor_same_label_ratio
        self.random_seed = random_seed
        self.instruction_template = instruction_template
        self.chunk_db: dict = {}
        self.label_to_chunks: dict = {}

    def _find_oracle_chunks(self, sample_refs: dict) -> list[str]:
        """Find oracle chunk IDs matching a sample's extracted references."""
        oracle_chunks = []

        for statute in sample_refs.get('statutes', []):
            law_name = statute['law_name']
            article = statute['article']
            for chunk_id, chunk in self.chunk_db.items():
                if chunk['source_type'] in ('statute', 'statute_fallback'):
                    if chunk['source_name'] == law_name and article in chunk.get('article', ''):
                        oracle_chunks.append(chunk_id)
                        break
            else:
                for chunk_id, chunk in self.chunk_db.items():
                    if chunk['source_type'] in ('statute', 'statute_fallback'):
                        if chunk['source_name'] == law_name:
                            oracle_chunks.append(chunk_id)
                            break

        for case in sample_refs.get('cases', []):
            case_number = case['case_number']
            for chunk_id, chunk in self.chunk_db.items():
                if chunk['source_type'] in ('case', 'case_fallback'):
                    if chunk['source_name'] == case_number:
                        oracle_chunks.append(chunk_id)
                        break

        seen = set()
        unique = []
        for cid in oracle_chunks:
            if cid not in seen:
                seen.add(cid)
                unique.append(cid)
        return unique[:self.max_oracle_docs]

    def _select_distractors(self, oracle_chunk_ids: list[str], major_label: str,
                            num_needed: int) -> list[str]:
        """Select distractor chunks: mix of same and different major_label."""
        oracle_set = set(oracle_chunk_ids)
        same_count = round(num_needed * self.distractor_same_label_ratio)
        distractors = []

        same_pool = [cid for cid in self.label_to_chunks.get(major_label, [])
                     if cid not in oracle_set]
        if same_pool:
            distractors.extend(random.sample(same_pool, min(same_count, len(same_pool))))

        diff_pool = []
        for label, cids in self.label_to_chunks.items():
            if label != major_label:
                diff_pool.extend([cid for cid in cids if cid not in oracle_set])
        if diff_pool:
            remaining = num_needed - len(distractors)
            distractors.extend(random.sample(diff_pool, min(remaining, len(diff_pool))))

        if len(distractors) < num_needed:
            all_pool = [cid for cid in self.chunk_db if cid not in oracle_set
                        and cid not in distractors]
            remaining = num_needed - len(distractors)
            if all_pool:
                distractors.extend(random.sample(all_pool, min(remaining, len(all_pool))))

        return distractors[:num_needed]

    def _build_label_to_chunks(self, ref_data: dict):
        """Map major_label -> list of chunk_ids."""
        self.label_to_chunks = {}
        for result in ref_data.get('results', []):
            label = result['major_label']
            if label not in self.label_to_chunks:
                self.label_to_chunks[label] = []
            for statute in result.get('statutes', []):
                for chunk_id, chunk in self.chunk_db.items():
                    if chunk['source_name'] == statute['law_name']:
                        self.label_to_chunks[label].append(chunk_id)
            for case in result.get('cases', []):
                for chunk_id, chunk in self.chunk_db.items():
                    if chunk['source_name'] == case['case_number']:
                        self.label_to_chunks[label].append(chunk_id)

        for label in self.label_to_chunks:
            self.label_to_chunks[label] = list(set(self.label_to_chunks[label]))

    @staticmethod
    def _format_doc_context(doc_pairs: list[tuple]) -> str:
        """Format documents as [D1] ..., [D2] ... text block."""
        lines = []
        for i, (_, chunk) in enumerate(doc_pairs, 1):
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

    @staticmethod
    def _extract_quote(chunk_text: str, max_len: int = 150) -> str:
        """Extract a concise quotable snippet from chunk text."""
        text = re.sub(r'\s+', ' ', chunk_text.strip())
        if len(text) <= max_len:
            return text
        cut = text[:max_len]
        m = list(re.finditer(r'[다요됨임함니까]\.|[.!?](?=\s|$)', cut))
        if m and m[-1].end() > 40:
            return cut[:m[-1].end()]
        return cut + "..."

    def _build_cot_citations(self, oracle_doc_map: dict, doc_pairs: list[tuple],
                             oracle_set: set) -> str:
        """Build Chain-of-Thought citation block from oracle documents."""
        citations = []
        for di, (cid, chunk) in enumerate(doc_pairs, 1):
            if cid not in oracle_set:
                continue
            quote = self._extract_quote(chunk['text'])
            source = chunk['source_name']
            article = chunk.get('article', '')
            if chunk['source_type'] in ('statute', 'statute_fallback'):
                header = f"{source} {article}".strip()
            else:
                court = chunk.get('court', '대법원')
                header = f"{court} {source}"
            citations.append(f"[D{di}]에 따르면, \"{header}: {quote}\"")
        return "\n".join(citations)

    def _reformat_answer(self, original_answer: str, oracle_doc_map: dict,
                         sample_refs: dict, doc_pairs: list[tuple], oracle_set: set) -> str:
        """Transform answer to CoT citation format with [Dn] references."""
        answer = original_answer.strip()
        answer = re.sub(r'「[^」]{80,}」', '', answer)
        answer = re.sub(r'"[^"]{80,}"', '', answer)
        answer = re.sub(r'\u201c[^\u201d]{80,}\u201d', '', answer)

        for statute in sample_refs.get('statutes', []):
            if statute['full_ref'] in oracle_doc_map:
                dn = f"[D{oracle_doc_map[statute['full_ref']]}]"
                answer = answer.replace(statute['full_ref'], dn, 1)

        for case in sample_refs.get('cases', []):
            ref_key = case['case_number']
            if ref_key in oracle_doc_map:
                dn = f"[D{oracle_doc_map[ref_key]}]"
                answer = answer.replace(case['full_ref'], dn, 1)

        answer = re.sub(r'\s+', ' ', answer).strip()
        if len(answer) > 500:
            cut = answer[:500]
            m = list(re.finditer(r'[다요됨임함니까]\.|[.!?](?=\s|$)', cut))
            if m and m[-1].end() > 100:
                answer = cut[:m[-1].end()]
            else:
                answer = cut

        cot_block = self._build_cot_citations(oracle_doc_map, doc_pairs, oracle_set)

        used_docs = sorted(set(re.findall(r'\[D\d+\]', answer)))
        if not used_docs and oracle_doc_map:
            doc_nums = sorted(set(oracle_doc_map.values()))
            used_docs = [f"[D{n}]" for n in doc_nums]
            answer += " (" + ", ".join(used_docs) + " 참조)"

        refs_section = "\n\n참고문헌:"
        for statute in sample_refs.get('statutes', []):
            refs_section += f"\n- {statute['full_ref']}"
        for case in sample_refs.get('cases', []):
            refs_section += f"\n- {case['court']} {case['case_number']} {case['judgment_type']}"

        if cot_block:
            return f"{cot_block}\n\n위 근거에 따르면, {answer}{refs_section}"
        return f"{answer}{refs_section}"

    def build_raft_dataset(self, df: pd.DataFrame, split_name: str,
                           ref_results: list[dict]) -> dict:
        """Build RAFT format dataset for one split."""
        random.seed(self.random_seed)
        ref_lookup = {r['index']: r for r in ref_results if r['split'] == split_name}

        instructions, inputs, outputs = [], [], []
        skipped = 0

        for idx, row in df.iterrows():
            sample_refs = ref_lookup.get(idx, {})
            major_label = row['major_label']
            oracle_chunk_ids = self._find_oracle_chunks(sample_refs)

            num_distractors = self.docs_per_sample - len(oracle_chunk_ids)
            if num_distractors < 0:
                oracle_chunk_ids = oracle_chunk_ids[:self.docs_per_sample]
                num_distractors = 0

            distractor_ids = self._select_distractors(oracle_chunk_ids, major_label, num_distractors)
            all_doc_ids = oracle_chunk_ids + distractor_ids
            oracle_set = set(oracle_chunk_ids)

            if len(all_doc_ids) < self.docs_per_sample:
                fill_pool = [cid for cid in self.chunk_db if cid not in set(all_doc_ids)]
                remaining = self.docs_per_sample - len(all_doc_ids)
                if fill_pool:
                    all_doc_ids.extend(random.sample(fill_pool, min(remaining, len(fill_pool))))

            random.shuffle(all_doc_ids)
            doc_pairs = [(cid, self.chunk_db[cid]) for cid in all_doc_ids if cid in self.chunk_db]

            if not doc_pairs:
                skipped += 1
                continue

            doc_context = self._format_doc_context(doc_pairs)
            oracle_doc_map = {}
            for di, (cid, chunk) in enumerate(doc_pairs, 1):
                if cid in oracle_set:
                    if 'statute' in chunk['source_type']:
                        key = f"{chunk['source_name']} {chunk.get('article', '')}".strip()
                        oracle_doc_map[key] = di
                    else:
                        oracle_doc_map[chunk['source_name']] = di

            reformatted = self._reformat_answer(
                row['answer'], oracle_doc_map, sample_refs, doc_pairs, oracle_set
            )
            instructions.append(f"{self.instruction_template}\n\n{doc_context}")
            inputs.append(row['question'])
            outputs.append(reformatted)

        if skipped > 0:
            logger.warning("Skipped %d samples (no chunks available)", skipped)

        return {"instruction": instructions, "input": inputs, "output": outputs}

    def run(self):
        """Execute RAFT dataset assembly."""
        with open(self.reference_extraction, 'r', encoding='utf-8') as f:
            ref_data = json.load(f)
        with open(self.chunk_database_path, 'r', encoding='utf-8') as f:
            self.chunk_db = json.load(f)

        logger.info("Loaded %d chunks", len(self.chunk_db))
        self._build_label_to_chunks(ref_data)
        ref_results = ref_data['results']

        for split_name, sampled_path in self.sampled_paths.items():
            df = pd.read_csv(sampled_path, dtype=str)
            output_path = self.dataset_paths[split_name]

            logger.info("Building %s dataset (%d samples)...", split_name, len(df))
            dataset = self.build_raft_dataset(df, split_name, ref_results)
            logger.info("Output: %d samples", len(dataset['instruction']))

            self.processed_data.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for i in range(len(dataset['instruction'])):
                    row = {
                        'instruction': dataset['instruction'][i],
                        'input': dataset['input'][i],
                        'output': dataset['output'][i],
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
            logger.info("Saved: %s", output_path)
