"""Step 3: Collect statute and case texts from the Korean National Law Information Center API."""

import hashlib
import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

_ABBREV_MAP = {
    '공익사업법': '공익사업을 위한 토지 등의 취득 및 보상에 관한 법률',
    '독점규제법': '독점규제 및 공정거래에 관한 법률',
    '산림법': '산림자원의 조성 및 관리에 관한 법률',
    '토지수용법': '공익사업을 위한 토지 등의 취득 및 보상에 관한 법률',
    '남녀고용평등법': '남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률',
    '도시정비법': '도시 및 주거환경정비법',
    '남북가족특례법': '남북 주민 사이의 가족관계와 상속 등에 관한 특례법',
    '진동관리법': '소음ㆍ진동관리법',
    '집시법': '집회 및 시위에 관한 법률',
    '특가법': '특정범죄 가중처벌 등에 관한 법률',
    '부동산실명법': '부동산 실권리자명의 등기에 관한 법률',
    '채무자회생법': '채무자 회생 및 파산에 관한 법률',
    '행형법': '형의 집행 및 수용자의 처우에 관한 법률',
    '노동관계조정법': '노동조합 및 노동관계조정법',
    '산재보험법': '산업재해보상보험법',
    '자동차손해배상법': '자동차손해배상 보장법',
    '파산법': '채무자 회생 및 파산에 관한 법률',
    '호적법': '가족관계의 등록 등에 관한 법률',
    '회사정리법': '채무자 회생 및 파산에 관한 법률',
    '성폭력처벌법': '성폭력범죄의 처벌 등에 관한 특례법',
    '성폭력특례법': '성폭력범죄의 처벌 등에 관한 특례법',
    '손해배상 보장법': '자동차손해배상 보장법',
    '예우등에관한법률시행령': '국가유공자 등 예우 및 지원에 관한 법률 시행령',
    '의소송비용산입에관한규칙': '변호사보수의 소송비용 산입에 관한 규칙',
    '중등교육법': '초ㆍ중등교육법',
    '토지보상법': '공익사업을 위한 토지 등의 취득 및 보상에 관한 법률',
    '세월호피해지원법': '4ㆍ16세월호참사 피해구제 및 지원 등을 위한 특별법',
    '재외동포법': '재외동포의 출입국과 법적 지위에 관한 법률',
    '헌법': '대한민국헌법',
    '노동조합법': '노동조합 및 노동관계조정법',
    '가정폭력처벌법': '가정폭력범죄의 처벌 등에 관한 특례법',
    '가정폭력특례법': '가정폭력범죄의 처벌 등에 관한 특례법',
    '동산채권담보법': '동산ㆍ채권 등의 담보에 관한 법률',
    '부동산중개업법': '공인중개사법',
    '사무관리규정': '행정업무의 운영 및 혁신에 관한 규정',
    '인지첩부법': '인지 첩부·첨부 및 공탁 제공에 관한 특례법',
    '임대주택법': '민간임대주택에 관한 특별법',
    '건물임대차보호법': '상가건물 임대차보호법',
}


class LawApiCollector:
    """Collect statute and case texts from the Korean law API.

    Encapsulates session and API credentials to eliminate parameter threading.
    """

    BASE_URL = "http://www.law.go.kr/DRF"

    def __init__(
        self,
        reference_extraction: Path,
        external_statutes: Path,
        external_cases: Path,
        api_delay: float = 0.5,
    ):
        self.reference_extraction = reference_extraction
        self.external_statutes = external_statutes
        self.external_cases = external_cases
        self.api_delay = api_delay
        self.session = self._create_session()
        self.oc = self._resolve_oc()

    @staticmethod
    def _create_session() -> requests.Session:
        """Create HTTP session with retry logic."""
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount("http://", HTTPAdapter(max_retries=retries))
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    @staticmethod
    def _resolve_oc() -> str:
        """Get the OC (user ID) from environment."""
        oc = os.environ.get("LAW_API_OC", "")
        if not oc:
            raise ValueError(
                "LAW_API_OC not set. Register at https://open.law.go.kr and set "
                "the LAW_API_OC environment variable."
            )
        return oc

    @staticmethod
    def _resolve_name(law_name: str) -> str:
        """Resolve abbreviation to a full searchable name."""
        if law_name in _ABBREV_MAP:
            return _ABBREV_MAP[law_name]
        for suffix in (' 시행령', ' 시행규칙', '시행령', '시행규칙'):
            if law_name.endswith(suffix):
                base = law_name[:-len(suffix)]
                if base in _ABBREV_MAP:
                    return _ABBREV_MAP[base] + ' ' + suffix.strip()
        return law_name

    def search_statute(self, law_name: str) -> str | None:
        """Search for a statute by name, return its serial number."""
        result = self._search_statute_once(law_name)
        if result is None:
            resolved = self._resolve_name(law_name)
            if resolved != law_name:
                result = self._search_statute_once(resolved)
        return result

    def _search_statute_once(self, law_name: str) -> str | None:
        """Single search attempt for a statute by name."""
        params = {
            "OC": self.oc, "target": "law", "type": "XML",
            "query": law_name, "display": "100",
        }
        try:
            resp = self.session.get(f"{self.BASE_URL}/lawSearch.do", params=params, timeout=30)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            query_norm = law_name.replace(' ', '')

            for item in root.findall(".//law"):
                name_elem = item.find("법령명한글")
                if name_elem is not None and name_elem.text:
                    name = name_elem.text.strip()
                    if name == law_name or name.replace(' ', '') == query_norm:
                        serial = item.find("법령일련번호")
                        if serial is not None:
                            return serial.text

            candidates = []
            for item in root.findall(".//law"):
                name_elem = item.find("법령명한글")
                if name_elem is not None and name_elem.text:
                    name = name_elem.text.strip()
                    idx = name.find(law_name)
                    if idx >= 0:
                        start_ok = (idx == 0 or name[idx - 1] in ' ㆍ·「(')
                        end_idx = idx + len(law_name)
                        end_ok = (end_idx == len(name) or name[end_idx] in ' ㆍ·」)')
                        if start_ok and end_ok:
                            serial = item.find("법령일련번호")
                            if serial is not None:
                                candidates.append((len(name), serial.text))
            if candidates:
                candidates.sort()
                return candidates[0][1]
        except Exception as e:
            logger.warning("Statute search failed for '%s': %s", law_name, e)
        return None

    def fetch_statute_text(self, serial_number: str) -> dict | None:
        """Fetch full statute text by serial number."""
        params = {"OC": self.oc, "target": "law", "MST": serial_number, "type": "XML"}
        try:
            resp = self.session.get(f"{self.BASE_URL}/lawService.do", params=params, timeout=60)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)

            name_elem = root.find(".//법령명_한글")
            law_name = (name_elem.text or "").strip() if name_elem is not None else ""

            articles = {}
            for article_elem in root.findall(".//조문단위"):
                article_no = article_elem.find("조문번호")
                if article_no is None or not article_no.text:
                    continue
                branch = article_elem.find("조문가지번호")
                key = f"제{article_no.text.strip()}조"
                if branch is not None and branch.text and branch.text.strip():
                    key += f"의{branch.text.strip()}"

                article_content = article_elem.find("조문내용")
                text = (article_content.text or "").strip() if article_content is not None else ""

                paragraphs = {}
                for para in article_elem.findall("항"):
                    para_no = para.find("항번호")
                    para_content = para.find("항내용")
                    if para_no is None or para_content is None:
                        continue
                    p_key = para_no.text.strip() if para_no.text else ""
                    p_text = (para_content.text or "").strip()
                    items = []
                    for ho in para.findall("호"):
                        ho_content = ho.find("호내용")
                        if ho_content is not None and ho_content.text:
                            items.append(ho_content.text.strip())
                    paragraphs[p_key] = {"content": p_text, "items": items}

                articles[key] = {"content": text, "paragraphs": paragraphs}

            return {"law_name": law_name, "serial": serial_number, "articles": articles}
        except Exception as e:
            logger.warning("Statute fetch failed for serial %s: %s", serial_number, e)
        return None

    def collect_statutes(self, unique_law_names: list[str]) -> dict:
        """Collect all statutes with file-based caching."""
        self.external_statutes.mkdir(parents=True, exist_ok=True)
        results = {}

        for i, law_name in enumerate(unique_law_names):
            safe_name = law_name.replace(' ', '_')
            if len(safe_name.encode('utf-8')) > 200:
                safe_name = hashlib.md5(law_name.encode()).hexdigest()
            cache_file = self.external_statutes / f"{safe_name}.json"

            if cache_file.exists():
                logger.info("[%d/%d] Cached: %s", i + 1, len(unique_law_names), law_name)
                with open(cache_file, 'r', encoding='utf-8') as f:
                    results[law_name] = json.load(f)
                continue

            logger.info("[%d/%d] Fetching: %s", i + 1, len(unique_law_names), law_name)
            serial = self.search_statute(law_name)
            if serial is None:
                results[law_name] = {"error": "not_found", "law_name": law_name}
            else:
                time.sleep(self.api_delay)
                data = self.fetch_statute_text(serial)
                results[law_name] = data if data else {
                    "error": "fetch_failed", "law_name": law_name, "serial": serial
                }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(results[law_name], f, ensure_ascii=False, indent=2)
            time.sleep(self.api_delay)

        return results

    def search_case(self, case_number: str) -> str | None:
        """Search for a court case by case number."""
        params = {
            "OC": self.oc, "target": "prec", "type": "XML",
            "query": case_number, "display": "5",
        }
        try:
            resp = self.session.get(f"{self.BASE_URL}/lawSearch.do", params=params, timeout=30)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)

            for item in root.findall(".//prec"):
                num_elem = item.find("사건번호")
                if num_elem is not None and case_number in (num_elem.text or ""):
                    serial = item.find("판례일련번호")
                    if serial is not None:
                        return serial.text
            first = root.find(".//판례일련번호")
            if first is not None:
                return first.text
        except Exception as e:
            logger.warning("Case search failed for '%s': %s", case_number, e)
        return None

    def fetch_case_text(self, serial_number: str) -> dict | None:
        """Fetch full case text by serial number."""
        params = {"OC": self.oc, "target": "prec", "ID": serial_number, "type": "XML"}
        try:
            resp = self.session.get(f"{self.BASE_URL}/lawService.do", params=params, timeout=60)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)

            def get_text(tag):
                elem = root.find(f".//{tag}")
                return elem.text.strip() if elem is not None and elem.text else ""

            return {
                "serial": serial_number,
                "case_name": get_text("사건명"),
                "case_number": get_text("사건번호"),
                "judgment_date": get_text("선고일자"),
                "court": get_text("법원명"),
                "judgment_type": get_text("사건종류명"),
                "holdings": get_text("판시사항"),
                "summary": get_text("판결요지"),
                "full_text": get_text("판례내용"),
                "referenced_statutes": get_text("참조조문"),
                "referenced_cases": get_text("참조판례"),
            }
        except Exception as e:
            logger.warning("Case fetch failed for serial %s: %s", serial_number, e)
        return None

    def collect_cases(self, unique_case_numbers: list[str]) -> dict:
        """Collect all cases with file-based caching."""
        self.external_cases.mkdir(parents=True, exist_ok=True)
        results = {}

        for i, case_number in enumerate(unique_case_numbers):
            cache_file = self.external_cases / f"{case_number}.json"
            if cache_file.exists():
                logger.info("[%d/%d] Cached: %s", i + 1, len(unique_case_numbers), case_number)
                with open(cache_file, 'r', encoding='utf-8') as f:
                    results[case_number] = json.load(f)
                continue

            logger.info("[%d/%d] Fetching: %s", i + 1, len(unique_case_numbers), case_number)
            serial = self.search_case(case_number)
            if serial is None:
                results[case_number] = {"error": "not_found", "case_number": case_number}
            else:
                time.sleep(self.api_delay)
                data = self.fetch_case_text(serial)
                results[case_number] = data if data else {
                    "error": "fetch_failed", "case_number": case_number, "serial": serial
                }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(results[case_number], f, ensure_ascii=False, indent=2)
            time.sleep(self.api_delay)

        return results

    def run(self) -> tuple:
        """Execute data collection pipeline."""
        with open(self.reference_extraction, 'r', encoding='utf-8') as f:
            ref_data = json.load(f)

        summary = ref_data['summary']

        logger.info("Statutes to collect: %d law names", summary['total_unique_law_names'])
        statute_results = self.collect_statutes(summary['unique_law_names'])
        success = sum(1 for v in statute_results.values() if 'error' not in v)
        logger.info("Statutes collected: %d/%d", success, len(statute_results))

        logger.info("Cases to collect: %d case numbers", summary['total_unique_cases'])
        case_results = self.collect_cases(summary['unique_case_numbers'])
        success = sum(1 for v in case_results.values() if 'error' not in v)
        logger.info("Cases collected: %d/%d", success, len(case_results))

        return statute_results, case_results
