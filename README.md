# Multi-Node LLaMA 13B Training

goguma3 + goguma4 이종 GPU 환경에서 LLaMA 13B를 학습합니다.

## 환경 정보

| 노드 | IP | GPU | VRAM | 역할 |
|------|-----|-----|------|------|
| goguma3 | 143.248.53.131 | 4x RTX 3060 Ti | 8GB x 4 | 마스터 |
| goguma4 | 143.248.53.206 | 2x RTX 2080 Ti | 11GB x 2 | 워커 |

총 6개 GPU 사용

## 사전 준비

### 1. 양쪽 노드에 환경 설치

```bash
# goguma3, goguma4 둘 다
conda activate llm_training
pip install torch transformers accelerate datasets peft bitsandbytes
DS_BUILD_OPS=0 pip install deepspeed
```

### 2. 코드 복사

```bash
# goguma3에서
cd ~/cs540
# 이 폴더의 파일들을 ~/cs540에 복사

# goguma4에도 동일하게 복사
scp -r ~/cs540/* goguma4:~/cs540/
```

### 3. Hugging Face 로그인 (양쪽 노드)

```bash
huggingface-cli login
```

### 4. 네트워크 인터페이스 확인

```bash
ip addr | grep "inet "
```

`eth0`이 아니면 `run_master.sh`, `run_worker.sh`에서 `NCCL_SOCKET_IFNAME` 수정

## 실행 방법

**중요: 두 노드에서 거의 동시에 실행해야 합니다!**

### 터미널 1 - goguma3 (마스터)

```bash
ssh goguma3
cd ~/cs540
conda activate llm_training
chmod +x run_master.sh
./run_master.sh
```

### 터미널 2 - goguma4 (워커)

```bash
ssh goguma4
cd ~/cs540
conda activate llm_training
chmod +x run_worker.sh
./run_worker.sh
```

마스터가 워커 연결을 기다리니까, 마스터 먼저 실행하고 바로 워커 실행하세요.

## 예상 출력

정상 시작되면 이런 로그가 나와요:

```
============================================
MASTER NODE: goguma3
MASTER_ADDR: 143.248.53.131
...
trainable params: 6,553,600 || all params: 6,738,415,616 || trainable%: 0.0973
...
{'loss': 2.xxx, 'learning_rate': xxx, 'epoch': xxx}
```

## 트러블슈팅

### 연결 안될 때

```bash
# 포트 열려있는지 확인
nc -zv 143.248.53.131 29500

# 방화벽 문제면 포트 변경
export MASTER_PORT=12345  # 둘 다 동일하게
```

### NCCL 에러

```bash
# 네트워크 인터페이스 확인
ifconfig
# 또는
ip addr

# 결과에서 143.248.x.x 가진 인터페이스 이름 확인 (예: enp0s31f6)
# run_master.sh, run_worker.sh에서 수정:
export NCCL_SOCKET_IFNAME=enp0s31f6
```

### OOM (메모리 부족)

`train_llama.py`에서:
```python
max_seq_length: int = 256  # 512 → 256으로 줄이기
lora_r: int = 8  # 16 → 8로 줄이기
```

### 타임아웃

```bash
# 둘 다 동시에 실행했는지 확인
# 30초 내로 둘 다 시작해야 함
```

## 파일 구조

```
.
├── train_llama.py      # 메인 학습 코드
├── ds_config.json      # DeepSpeed 설정
├── run_master.sh       # goguma3용 실행 스크립트
├── run_worker.sh       # goguma4용 실행 스크립트
└── README.md
```
