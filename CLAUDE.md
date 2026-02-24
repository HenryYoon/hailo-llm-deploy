# CLAUDE.md — hailo-llm-deploy

## 프로젝트 개요

- **레포**: https://github.com/HenryYoon/legal-chatbot
- **현재 상태**: trial2 → main merge 완료. 1 star, 4 commits, Apache-2.0
- **목표**: HuggingFace 소형 LLM을 Hailo-10H NPU에 배포하는 범용 CLI 파이프라인 도구로 리브랜딩
- **배경**: 기존 한국 법률 AI 레포(RAFT 파인튜닝)에서 도메인 로직을 분리하고 범용화

---

## 현재 코드베이스 분석 (trial2 merge 후)

### README 기록 사항 (main 브랜치)

- ONNX → HAR 컴파일 실패 기록됨 (Hailo DFC 단계에서 막힘)
- TODO에 RAFT 데이터 구축, Qwen2-1.5B LoRA, LoRA + pre-compiled HAR 결합 항목 존재
- trial3 브랜치가 local only로 존재하므로 이 TODO 진행 상태 확인 필요

### 모델 체크포인트 현황 (gitignored)

| 경로 | 내용 |
|------|------|
| models/merged/trial1 | Qwen 3B, Alpaca format, 16bit merged |
| models/merged/trial2 | Qwen 3B(?), ChatML |
| models/merged/trial2.1 | Qwen 1.5B, ChatML (train_trial2.py의 MODEL_NAME과 일치) |
| models/checkpoints/trial2 | checkpoint-50, 100, 125 (조기 중단으로 보임) |
| models/checkpoints/trial2.1 | checkpoint-750, 1200, 1250 (10 epoch × 125 steps/epoch = 정상 완료) |
| models/lora_adapters/stage1 | trial1 LoRA |
| models/lora_adapters/trial2 | |
| models/lora_adapters/trial2.1 | |

**관찰:** trial2 체크포인트가 125에서 멈춘 것으로 보아 1 epoch 학습 후 trial2.1로 전환한 것으로 추정.
trial2.1은 1250 체크포인트까지 존재하며 save_steps=50 기준 정상 완료.

### 전체 디렉토리 구조 (gitignore 포함)

```
legal-chatbot/
├── data/
│   ├── raw/                          # 원본 CSV (law_qa_v1.csv 등)
│   ├── processed/
│   │   ├── metadata/                 # 샘플링 통계, 참조 추출 결과
│   │   └── trial1/                   # trial1 데이터셋
│   └── external/
│       ├── statutes/                 # 국가법령정보센터 API 캐시 (JSON)
│       └── cases/                    # 판례 API 캐시 (JSON)
├── src/
│   ├── config.py                     # 🔴 중앙 설정 (제공 안 됨 — 확인 필요)
│   ├── construct_trial1.py           # [Legacy] Trial 1 데이터 전처리
│   ├── train_stage1.py               # [Legacy] Trial 1 학습 (Alpaca, Qwen 3B)
│   ├── construct_trial2.py           # Trial 2 RAFT 파이프라인 진입점
│   ├── sampler.py                    # Step 1: 층화 샘플링
│   ├── extractor.py                  # Step 2: 법령/판례 참조 추출
│   ├── collector.py                  # Step 3: 국가법령정보센터 API 수집
│   ├── chunker.py                    # Step 4: 문서 청킹
│   ├── raft_builder.py               # Step 5: RAFT 데이터셋 조립
│   ├── train_trial2.py               # Trial 2 학습 (ChatML, Qwen 1.5B)
│   ├── evaluate_trial2.py            # 평가
│   ├── convert_formal.py             # 후처리: 습니다체 변환
│   └── export_onnx.py                # ONNX 익스포트
├── models/                           # ⚠️ gitignored — 로컬에만 존재
│   ├── checkpoints/
│   │   ├── trial2/                   # checkpoint-50, 100, 125 + tensorboard
│   │   └── trial2.1/                 # checkpoint-750, 1200, 1250 + tensorboard
│   ├── lora_adapters/
│   │   ├── stage1/                   # trial1 LoRA
│   │   ├── trial2/
│   │   └── trial2.1/
│   └── merged/
│       ├── trial1/                   # Qwen 3B merged 16bit
│       ├── trial2/                   # Qwen 3B merged
│       └── trial2.1/                 # Qwen 1.5B merged
├── hailo_ai_sw_suite/                # ⚠️ gitignored — Hailo SDK 로컬 설치
│   ├── artifacts/
│   ├── docs/
│   ├── examples/
│   │   ├── c/                        # C API 예제 (vstreams, pipeline 등)
│   │   ├── cpp/                      # C++ API 예제 (async infer 등)
│   │   ├── genai/                    # 🔑 GenAI 예제 (chat, speech2text, vlm)
│   │   └── hefs/                     # 컴파일된 HEF 파일
│   ├── sources/
│   │   └── model_zoo/                # Hailo Model Zoo 전체 소스
│   │       ├── hailo_model_zoo/      # 핵심: cfg, core, postprocessing 등
│   │       ├── hailo_models/         # Hailo 커스텀 모델 (LPR, ReID 등)
│   │       └── training/             # YOLO 시리즈, ViT 등 학습 스크립트
│   └── tappas/
│       └── detection/                # TAPPAS 파이프라인 (h10 리소스)
├── infra/                            # ⚠️ gitignored — 인프라 설정
│   ├── config/
│   ├── docker/
│   └── hailo/
├── notebooks/
├── results/                          # 평가 결과 (eval_trialN.json)
├── logs/
├── docs/
│   └── result_trial1.png
├── unsloth_compiled_cache/           # ⚠️ gitignored — Unsloth 컴파일 캐시
├── .gitignore
├── env_legal.yml                     # Conda 환경
├── LICENCE.md
└── README.md
```

### 핵심 발견: hailo_ai_sw_suite 로컬 존재

Hailo SDK가 로컬에 전체 설치되어 있다. 특히:
- `examples/genai/`: chat_example, speech2text_example, vlm_example — Phase 3에서 직접 참조 가능
- `sources/model_zoo/`: DFC 설정 파일(alls/hailo10h/), 후처리 코드 전체 보유
- `tappas/detection/resources/h10`: Hailo-10H용 TAPPAS 리소스

이는 Phase 3 Hailo 통합의 진입 장벽을 크게 낮춘다.

### infra/ 디렉토리

config, docker, hailo 서브폴더가 이미 존재한다.
리팩토링 시 새로 만들 필요 없이 기존 구조를 활용.

### 코드 품질 평가

**잘 된 부분 (trial2 코드):**
- construct_trial2.py: argparse 기반 단계별 실행, 모듈 분리 깔끔
- collector.py: 캐싱, retry 로직, 약어 사전(_ABBREV_MAP) 등 실무적
- extractor.py: 한국 법률 인용 regex가 상당히 정교함 (상대 참조 해석 포함)
- train_trial2.py: ChatML 포맷 사용 (Qwen 네이티브), Path 객체 사용, 하이퍼파라미터 정리됨
- evaluate_trial2.py: auto-discover trials, 다중 trial 비교 테이블, CLI 인터페이스

**수정 필요 사항:**

| 파일 | 이슈 | 심각도 |
|------|------|--------|
| config.py | 제공되지 않음 — 모든 모듈이 import하므로 반드시 확인 필요 | 🔴 Critical |
| export_onnx.py | 경로 `../../models/merged/stage1_16bit` 하드코딩. trial2 모델 미지원 | 🔴 High |
| export_onnx.py | model 로드 후 미사용, main_export에 경로만 전달 (불필요한 VRAM 점유) | 🟡 Medium |
| raft_builder.py | `_to_formal_style()` 함수 정의되어 있으나 어디서도 호출 안 됨 (dead code). convert_formal.py가 LLM으로 대체 | 🟡 Medium |
| train_stage1.py | 주석 "Qwen2.5-7B" vs 실제 로드 "3B" 불일치 | 🟡 Medium |
| train_stage1.py | `save_strategy='best'`는 유효하지 않은 SFTConfig 값 | 🟡 Medium |
| train_stage1.py | `# %%` 셀 구분자, 이모지 로깅, model/tokenizer가 if __name__ 바깥에서 로드됨 | 🟡 Medium |
| construct_trial1.py | `data2_train` 선언 후 `data2` 전체를 순회 (버그) | 🟡 Medium |
| construct_trial1.py | `# %%` 셀 구분자, 하드코딩 경로, Unsloth 추론 코드 혼재 | 🟡 Medium |
| train_trial2.py | trial2.1 디렉토리명에 모델이 1.5B인데 README는 "Qwen2.5-3B" 기재. trial2→trial2.1 전환 경위 불명확 | 🟢 Low |

---

## 리팩토링 계획

### Phase 0: 정리 (3일)

trial1 레거시 코드를 정리하고 config.py를 확인한다.

- `construct_trial1.py`, `train_stage1.py` → `legacy/` 디렉토리로 이동
- `export_onnx.py` → trial2 모델 경로 지원하도록 수정, 불필요한 model 로드 제거
- `raft_builder.py` → `_to_formal_style()` dead code 제거
- `config.py` 내용 확인 후 문서화

### Phase 1: 구조 분리 (1주)

도메인(법률) 코드와 범용 파이프라인 코드를 분리한다.
기존 `infra/`, `hailo_ai_sw_suite/` 구조를 최대한 활용.

```
# 목표 구조 (변경분만 표시)
legal-chatbot/  →  hailo-llm-deploy/  (최종 rename)
├── hailo_llm_deploy/                 # 🆕 범용 파이프라인 패키지
│   ├── __init__.py
│   ├── cli.py                        # Typer CLI 진입점
│   ├── config.py                     # Pydantic 설정 모델 (YAML)
│   ├── finetune.py                   # ← train_trial2.py 범용화
│   ├── export.py                     # ← export_onnx.py 리팩토링
│   ├── quantize.py                   # INT8/INT4 양자화
│   ├── deploy.py                     # Hailo HEF 컴파일 + 추론 서버
│   ├── serve.py                      # FastAPI 추론 엔드포인트
│   └── evaluate.py                   # ← evaluate_trial2.py 범용화
├── configs/                          # 🆕
│   ├── default.yaml
│   └── examples/
│       └── korean_legal.yaml
├── examples/                         # 🆕
│   └── korean-legal/                 # ← src/ 도메인 코드 이동
│       ├── construct.py              # ← construct_trial2.py
│       ├── sampler.py
│       ├── extractor.py
│       ├── collector.py
│       ├── chunker.py
│       ├── raft_builder.py
│       ├── convert_formal.py
│       ├── config.py                 # ← src/config.py (법률 도메인용)
│       └── README.md
├── src/                              # 🔄 legacy/ 로 이동할 파일만 남김
│   └── legacy/
│       ├── construct_trial1.py
│       └── train_stage1.py
├── infra/                            # ✅ 기존 유지
│   ├── config/
│   ├── docker/                       # deploy.py에서 참조
│   └── hailo/                        # Hailo 관련 설정
├── hailo_ai_sw_suite/                # ✅ 기존 유지 (gitignored)
├── models/                           # ✅ 기존 유지 (gitignored)
├── data/                             # ✅ 기존 유지
├── results/                          # ✅ 기존 유지
├── tests/                            # 🆕
├── pyproject.toml                    # 🆕
└── README.md                         # 🔄 전면 재작성
```

**이동 매핑:**

| 현재 위치 | 목표 위치 | 비고 |
|-----------|----------|------|
| src/train_trial2.py | hailo_llm_deploy/finetune.py | 법률 instruction 제거, YAML config 기반으로 |
| src/evaluate_trial2.py | hailo_llm_deploy/evaluate.py | 프롬프트 포맷 이미 파라미터화됨 |
| src/export_onnx.py | hailo_llm_deploy/export.py | 하드코딩 경로 제거, CLI 인자 |
| src/sampler.py | examples/korean-legal/ | 도메인 코드 |
| src/extractor.py | examples/korean-legal/ | 도메인 코드 |
| src/collector.py | examples/korean-legal/ | 도메인 코드 |
| src/chunker.py | examples/korean-legal/ | 도메인 코드 |
| src/raft_builder.py | examples/korean-legal/ | 도메인 코드 (dead code 제거 후) |
| src/convert_formal.py | examples/korean-legal/ | 도메인 코드 |
| src/construct_trial1.py | src/legacy/ | 레거시 보존 |
| src/train_stage1.py | src/legacy/ | 레거시 보존 |

**건드리지 않는 것:**
- `hailo_ai_sw_suite/` — SDK 로컬 설치. 그대로 둔다
- `infra/` — docker, hailo config 기존 구조 활용
- `models/` — 체크포인트, LoRA, merged 모델 구조 유지
- `data/` — raw, processed, external 구조 유지

### Phase 2: CLI 래핑 (1주)

```bash
hailo-llm-deploy finetune --config configs/examples/korean_legal.yaml
hailo-llm-deploy export --model ./my-model --format onnx --dtype float16
hailo-llm-deploy quantize --model ./model.onnx --target int8
hailo-llm-deploy evaluate --model ./my-model --test-data ./test.jsonl
hailo-llm-deploy serve --model ./model.hef --port 8000
```

라이브러리: `typer` + `rich` (CLI), `pydantic` + `pyyaml` (config), `fastapi` + `uvicorn` (서버)

### Phase 3: Hailo 파이프라인 통합 (2~3주)

**⚠️ 핵심 블로커: ONNX → HAR 컴파일이 trial1에서 실패한 이력 있음.**
Hailo DFC가 Transformer 모델 구조를 지원하는지 재확인 필요. Hailo-10H는 pre-compiled HEF만 지원하는 제약이 있을 수 있음 (hailo-ollama API 경유가 현실적 대안).

- `hailo_ai_sw_suite/examples/genai/chat_example` 참조하여 추론 서버 구현
- `hailo_ai_sw_suite/sources/model_zoo/` 내 DFC 설정(alls/hailo10h/) 활용
- `infra/docker/`에 기존 Docker 설정 존재 — 컨테이너화 시 활용
- `infra/hailo/`에 기존 Hailo 설정 존재 — deploy.py에서 참조
- RPi5 + AI HAT+ 원클릭 설정 스크립트
- 벤치마크 자동 측정 (토큰/초, 메모리, 전력)

### Phase 4: 문서화 + 런칭 (1주)

- README.md (GIF 데모, 비교 테이블, 원커맨드 설치)
- docs/ (Getting Started, Configuration, Supported Models)
- GitHub Actions CI, PyPI 배포

---

## 설정 파일 스키마 (configs/default.yaml)

```yaml
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  max_seq_length: 2048
  load_in_4bit: true

lora:
  r: 16
  alpha: 16
  dropout: 0
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

training:
  epochs: 10
  batch_size: 2
  gradient_accumulation: 4
  learning_rate: 2e-4
  warmup_ratio: 0.03
  scheduler: cosine
  weight_decay: 0.01
  eval_steps: 50
  save_steps: 50
  seed: 42
  prompt_format: chatml       # chatml | alpaca

data:
  train_path: null
  val_path: null
  test_path: null
  instruction: null           # 도메인별 system prompt

export:
  format: onnx
  dtype: float16
  output_dir: ./output

deploy:
  target: hailo-10h           # hailo-10h | hailo-8l | onnxruntime
  port: 8000

evaluate:
  metrics:
    - rouge_l
    - bertscore
  llm_judge: false
  prompt_format: chatml
```

---

## 코딩 컨벤션

- Python 3.10+, type hints 필수
- docstring: Google style
- 포매터: ruff
- 테스트: pytest
- 로깅: `logging` 모듈 (이모지 금지)
- 경로: `pathlib.Path` 사용 (train_trial2.py 스타일 따름)
- config: 모든 하이퍼파라미터는 YAML config 또는 CLI 인자. 코드 내 하드코딩 금지
- 에러: 사용자 대면 에러는 `rich` 패널 출력

---

## 브랜치 현황 및 전략

### 현재 브랜치 (git branch -a)

| 브랜치 | remote | 상태 |
|--------|--------|------|
| `main` ✱ | origin/main | trial2 merge 완료. 현재 작업 브랜치 |
| `trial1` | origin/trial1 | Legacy. Alpaca format, Qwen 3B, 16.5K 데이터 |
| `trial2` | origin/trial2 | RAFT 파이프라인, ChatML, Qwen 1.5B |
| `trial3` | ❌ (local only) | 미푸시. models/ 하위에 trial3 체크포인트 없음 — 작업 초기 또는 미시작 가능성 |

### 정리 계획

**Phase 0에서 처리:**
- `trial1`, `trial2` → 보존 (히스토리 참조용). 추가 작업 없음
- `trial3` → 내용 확인 후 main에 반영할 것이 있으면 merge, 없으면 삭제
- `trial3`이 remote에 없으므로 push 여부 결정 필요

**리팩토링 시작 후:**
- `main`: 안정 릴리스 (리팩토링 완료 코드만)
- `dev`: 개발 통합
- `refactor/phase0-cleanup`: 레거시 정리 (main에서 분기)
- `refactor/phase1-structure`: 구조 분리
- `feat/cli`: CLI 래핑
- `feat/hailo-pipeline`: Hailo 통합

**리팩토링 완료 후 (오픈소스 공개 시점):**
- `trial1`, `trial2`, `trial3` 브랜치 삭제
- 레포명 `legal-chatbot` → `hailo-llm-deploy`로 변경 (GitHub redirect 자동 생성됨)
- 또는 새 레포 생성 후 `legal-chatbot`은 archive

---

## Skills / Agent 정의

**현 단계에서 불필요.**

이 프로젝트는 CLI 도구다. 사용자가 명확한 커맨드를 입력하고 결정적 결과를 받는다.
LLM 에이전트를 끼워넣으면 복잡성만 늘고, 디버깅이 어려워지고, 의존성이 무거워진다.

**향후 검토 시점:**
- 1,000+ 스타 달성 후 확장 시
- "자연어로 배포 설정 기술 → 자동 구성" 기능 추가 시
- 그때도 MCP 서버로 기존 CLI를 래핑하는 것이 현실적