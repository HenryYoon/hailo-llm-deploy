"""Step 4: Chunk collected statute and case texts into RAFT-appropriate segments."""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Chunk legal documents at article granularity with sentence-boundary splitting."""

    def __init__(
        self,
        external_statutes: Path,
        external_cases: Path,
        reference_extraction: Path,
        chunk_database: Path,
        metadata_dir: Path,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ):
        self.external_statutes = external_statutes
        self.external_cases = external_cases
        self.reference_extraction = reference_extraction
        self.chunk_database = chunk_database
        self.metadata_dir = metadata_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def _extract_article_key(article_ref: str) -> str:
        """Extract the article-level key from a reference (e.g., '제246조 제1항' -> '제246조')."""
        match = re.match(r'(제\d+조(?:의\d+)?)', article_ref)
        return match.group(1) if match else article_ref

    @staticmethod
    def _format_article_text(article_data: dict) -> str:
        """Format article content + paragraphs into readable text."""
        content = article_data.get('content', '')
        paragraphs = article_data.get('paragraphs', {})
        parts = [content] if content else []

        for p_key in sorted(paragraphs.keys()):
            p_data = paragraphs[p_key]
            if p_data.get('content'):
                parts.append(p_data['content'])
            for item in p_data.get('items', []):
                parts.append(f"  {item}")

        return "\n".join(parts)

    def _get_referenced_articles(self, ref_data: dict) -> tuple:
        """Build mapping: law_name -> set of article keys from references."""
        law_articles = {}
        referenced_cases = set()

        for result in ref_data.get('results', []):
            for statute in result.get('statutes', []):
                law_name = statute['law_name']
                article_key = self._extract_article_key(statute['article'])
                if law_name not in law_articles:
                    law_articles[law_name] = set()
                law_articles[law_name].add(article_key)
            for case in result.get('cases', []):
                referenced_cases.add(case['case_number'])

        return law_articles, referenced_cases

    def chunk_by_sentences(self, text: str) -> list[str]:
        """Split text into overlapping chunks at Korean sentence boundaries."""
        if not text or not text.strip():
            return []

        sentences = re.split(r'(?<=[.다요함임됨])\s+|\n+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return [text.strip()] if text.strip() else []

        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 > self.chunk_size and current:
                chunks.append(current.strip())
                if self.chunk_overlap > 0 and len(current) > self.chunk_overlap:
                    current = current[-self.chunk_overlap:] + " " + sentence
                else:
                    current = sentence
            else:
                current = (current + " " + sentence).strip()
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def chunk_statute(self, law_name: str, statute_data: dict, referenced_keys: set) -> list[dict]:
        """Chunk a statute at article granularity for referenced articles only."""
        chunks = []
        if 'error' in statute_data:
            return chunks

        for article_key, article_data in statute_data.get('articles', {}).items():
            if article_key not in referenced_keys:
                continue
            full_text = self._format_article_text(article_data)
            if not full_text.strip():
                continue

            if len(full_text) <= self.chunk_size:
                chunks.append({
                    'chunk_id': f"statute_{law_name}_{article_key}",
                    'source_type': 'statute',
                    'source_name': law_name,
                    'article': article_key,
                    'text': full_text.strip(),
                })
            else:
                for ci, sc in enumerate(self.chunk_by_sentences(full_text)):
                    chunks.append({
                        'chunk_id': f"statute_{law_name}_{article_key}_c{ci}",
                        'source_type': 'statute',
                        'source_name': law_name,
                        'article': article_key,
                        'text': sc,
                    })
        return chunks

    def chunk_case(self, case_number: str, case_data: dict) -> list[dict]:
        """Chunk a court case into holdings/summary chunks."""
        chunks = []
        if 'error' in case_data:
            return chunks

        for section_name in ('summary', 'holdings'):
            section_text = case_data.get(section_name, '')
            if not section_text or not section_text.strip():
                continue
            section_text = section_text.replace('<br/>', '\n').replace('<br>', '\n')
            section_text = re.sub(r'<[^>]+>', '', section_text).strip()

            for ci, sc in enumerate(self.chunk_by_sentences(section_text)):
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

    def build_chunk_database(self, ref_data: dict) -> dict:
        """Build the chunk database from external data for referenced articles."""
        all_chunks = {}
        law_articles, referenced_cases = self._get_referenced_articles(ref_data)
        logger.info("Referenced: %d law names, %d case numbers",
                     len(law_articles), len(referenced_cases))

        for sf in self.external_statutes.glob("*.json"):
            with open(sf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'error' in data:
                continue
            stem_name = sf.stem.replace('_', ' ')
            referenced_keys = law_articles.get(stem_name)
            if referenced_keys is None:
                continue
            for chunk in self.chunk_statute(stem_name, data, referenced_keys):
                all_chunks[chunk['chunk_id']] = chunk

        for cf in self.external_cases.glob("*.json"):
            case_number = cf.stem
            if case_number not in referenced_cases:
                continue
            with open(cf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for chunk in self.chunk_case(case_number, data):
                all_chunks[chunk['chunk_id']] = chunk

        return all_chunks

    def build_fallback_chunks(self, ref_data: dict) -> dict:
        """Build fallback chunks for references that couldn't be fetched."""
        fallback = {}

        existing_statutes = {}
        for sf in self.external_statutes.glob("*.json"):
            with open(sf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'error' not in data:
                existing_statutes[sf.stem.replace('_', ' ')] = data

        existing_cases = set()
        for cf in self.external_cases.glob("*.json"):
            with open(cf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'error' not in data:
                existing_cases.add(cf.stem)

        for result in ref_data.get('results', []):
            for statute in result.get('statutes', []):
                law_name = statute['law_name']
                article_key = self._extract_article_key(statute['article'])
                if law_name in existing_statutes:
                    if article_key in existing_statutes[law_name].get('articles', {}):
                        continue
                full_ref = statute['full_ref']
                chunk_id = f"fallback_statute_{full_ref.replace(' ', '_')}"
                if chunk_id not in fallback:
                    fallback[chunk_id] = {
                        'chunk_id': chunk_id,
                        'source_type': 'statute_fallback',
                        'source_name': law_name,
                        'article': statute['article'],
                        'text': f"{full_ref} (원문 미수집)",
                    }

            for case in result.get('cases', []):
                if case['case_number'] not in existing_cases:
                    chunk_id = f"fallback_case_{case['case_number']}"
                    if chunk_id not in fallback:
                        fallback[chunk_id] = {
                            'chunk_id': chunk_id,
                            'source_type': 'case_fallback',
                            'source_name': case['case_number'],
                            'article': 'fallback',
                            'text': f"{case['full_ref']} (원문 미수집)",
                            'court': case.get('court', ''),
                        }

        return fallback

    def run(self) -> dict:
        """Execute chunking pipeline."""
        with open(self.reference_extraction, 'r', encoding='utf-8') as f:
            ref_data = json.load(f)

        all_chunks = self.build_chunk_database(ref_data)
        logger.info("External chunks: %d", len(all_chunks))

        fallback = self.build_fallback_chunks(ref_data)
        logger.info("Fallback chunks: %d", len(fallback))

        all_chunks.update(fallback)
        logger.info("Total chunks: %d", len(all_chunks))

        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        with open(self.chunk_database, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        logger.info("Saved to %s", self.chunk_database)
        return all_chunks
