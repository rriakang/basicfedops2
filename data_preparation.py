# data_preparation.py
# ——————————————————————————————————————————————————————————————
# Drop-in replacement: client_main.py 수정 없이 사용
# - 고정 설정: NUM_CLIENTS, PARTITION_STRATEGY, DIRICHLET_ALPHA, SPLIT_SEED
# - "그냥 ID만" 쓰고 싶으면 CLIENT_ID_OVERRIDE에 서버별 고정 ID(0..NUM_CLIENTS-1) 지정
# - CLIENT_ID_OVERRIDE=None 인 경우 파일락 기반 자동 배정(동일 머신/공유 디렉토리에서 유효)
# ——————————————————————————————————————————————————————————————

import os
import json
import logging
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import datasets, transforms

# (선택) 파일 락: 동일 머신에서 여러 프로세스 자동 배정용
try:
    import fcntl  # type: ignore
except Exception:
    fcntl = None

# ===== 고정 정책 (코드 내부에서만 제어) =====
NUM_CLIENTS        = 4             # 전체 클라이언트 수(모든 서버 동일)
PARTITION_STRATEGY = "dirichlet"   # "iid" 또는 "dirichlet"
DIRICHLET_ALPHA    = 0.1           # Non-IID 강도(작을수록 편향 ↑)
SPLIT_SEED         = 42            # 재현성 시드
VAL_SPLIT_DEFAULT  = 0.1           # 기본 validation 비율(인자 우선)
TEST_SPLIT_FIXED   = 0.2           # 기존 코드 유지

# === "그냥 ID만" 수동 배정 ===
# 각 서버에서 이 값만 다르게 설정하세요 (0..NUM_CLIENTS-1).
# 예) x서버: 0, y서버: 1, z서버: 2, w서버: 3
CLIENT_ID_OVERRIDE = None  # ← 여기만 서버별로 수정하면 끝. None이면 자동배정 사용.

# (선택) 자동 배정용 로컬 랜데부 디렉토리
RENDEZVOUS_DIR   = "./.flrendezvous"
ASSIGN_JSON_PATH = os.path.join(RENDEZVOUS_DIR, "assign.json")
ASSIGN_LOCK_PATH = os.path.join(RENDEZVOUS_DIR, "assign.lock")

# set log format
handlers_list = [logging.StreamHandler()]
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)8.8s] %(message)s",
    handlers=handlers_list,
)
logger = logging.getLogger(__name__)


# ---------------------------
# 내부 유틸
# ---------------------------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _pid_alive(pid: int) -> bool:
    # 리눅스/맥 기준 간단 체크
    return os.path.exists(f"/proc/{pid}")

def _flock_file(fd):
    if fcntl is None:
        return  # 락 생략(윈도우 등) — 동시 배정 충돌 가능
    fcntl.flock(fd, fcntl.LOCK_EX)

def _funlock_file(fd):
    if fcntl is None:
        return
    fcntl.flock(fd, fcntl.LOCK_UN)

def _assign_client_id_auto(num_clients: int) -> int:
    """
    파일락 + 로컬 레지스트리(assign.json)로 현재 프로세스에 고유 client_id 배정.
    (동일 머신 혹은 공유 디렉토리 사용 시 유효)
    """
    _ensure_dir(RENDEZVOUS_DIR)

    lock_fd = os.open(ASSIGN_LOCK_PATH, os.O_CREAT | os.O_RDWR)
    try:
        _flock_file(lock_fd)
        if os.path.isfile(ASSIGN_JSON_PATH):
            with open(ASSIGN_JSON_PATH, "r", encoding="utf-8") as f:
                try:
                    reg = json.load(f)
                except Exception:
                    reg = {}
        else:
            reg = {}

        # 죽은 PID 정리
        to_del = []
        for key, rec in list(reg.items()):
            if not isinstance(rec, dict):
                to_del.append(key); continue
            pid = rec.get("pid")
            if not isinstance(pid, int) or not _pid_alive(pid):
                to_del.append(key)
        for k in to_del:
            reg.pop(k, None)

        # 현재 프로세스 키
        host = os.uname().nodename if hasattr(os, "uname") else "host"
        proc_key = f"{host}:{os.getpid()}"
        if proc_key in reg:
            cid = int(reg[proc_key]["client_id"])
        else:
            used = set(int(v["client_id"]) for v in reg.values() if isinstance(v, dict) and "client_id" in v)
            candidates = [i for i in range(num_clients) if i not in used]
            if candidates:
                cid = candidates[0]
            else:
                cid = (max(used) + 1) % num_clients if used else 0
                logger.warning(f"[rendezvous] All IDs in use; assigning cyclic id={cid}")
            reg[proc_key] = {"pid": os.getpid(), "client_id": cid}

        with open(ASSIGN_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(reg, f, ensure_ascii=False, indent=2)

        return cid
    finally:
        _funlock_file(lock_fd)
        os.close(lock_fd)

def _get_targets_from_torchvision_dataset(ds):
    t = getattr(ds, "targets", None)
    if t is None:
        labels = []
        for i in range(len(ds)):
            _, y = ds[i]
            labels.append(int(y.item() if isinstance(y, torch.Tensor) else y))
        return labels
    if isinstance(t, torch.Tensor):
        t = t.tolist()
    return list(map(int, t))

def _dirichlet_partition_indices(targets, num_clients: int, alpha: float, seed: int):
    """클래스별 Dirichlet(alpha) 비율로 Non-IID 인덱스 분할."""
    rng = np.random.default_rng(seed)
    targets = np.array(targets)
    classes = np.unique(targets)
    client_indices = [[] for _ in range(num_clients)]

    for c in classes:
        cls_idx = np.where(targets == c)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)

        probs = rng.dirichlet([alpha] * num_clients)
        counts = np.floor(probs * n).astype(int)
        rest = n - counts.sum()
        if rest > 0:
            order = np.argsort(-probs)
            counts[order[:rest]] += 1

        start = 0
        for k in range(num_clients):
            cnt = counts[k]
            if cnt > 0:
                client_indices[k].extend(cls_idx[start:start + cnt].tolist())
                start += cnt

    for k in range(num_clients):
        rng.shuffle(client_indices[k])

    return client_indices


# ---------------------------
# Public API (기존 시그니처 유지)
# ---------------------------
def load_partition(dataset, validation_split, batch_size):
    """
    기존과 동일한 시그니처/반환값.
    - client_main.py, Hydra 설정, 환경변수 변경 없이 사용.
    - 서버별 고정 ID로 쓰고 싶으면 파일 상단 CLIENT_ID_OVERRIDE만 바꾸세요.
    """
    # 로그 (기존 유지)
    now = datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    fl_task = {"dataset": dataset, "start_execution_time": now_str}
    logging.info(f'FL_Task - {json.dumps(fl_task)}')

    # MNIST 전처리/로딩 (기존 유지)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Grayscale
    ])
    full_dataset = datasets.MNIST(root='./dataset/mnist', train=True, download=True, transform=transform)

    # —— ID 결정(우선순위: 수동 고정 → 자동 배정) ——
    num_clients = NUM_CLIENTS
    if CLIENT_ID_OVERRIDE is not None:
        client_id = int(CLIENT_ID_OVERRIDE)
        assert 0 <= client_id < num_clients, f"client_id {client_id} must be in [0,{num_clients-1}]"
        logging.info(f"[override] using fixed client_id={client_id}/{num_clients}")
    else:
        client_id = _assign_client_id_auto(num_clients)
        logging.info(f"[auto] assigned client_id={client_id}/{num_clients}")

    strategy   = PARTITION_STRATEGY.lower()
    alpha      = DIRICHLET_ALPHA
    split_seed = SPLIT_SEED

    # —— 클라이언트 shard 선택 ——
    base_dataset = full_dataset
    if num_clients > 1:
        if strategy == "dirichlet":
            targets = _get_targets_from_torchvision_dataset(full_dataset)
            shards = _dirichlet_partition_indices(targets, num_clients, alpha, split_seed)
            client_idx = shards[client_id]
            base_dataset = Subset(full_dataset, client_idx)
            logging.info(
                f"[NonIID-Dirichlet] clients={num_clients} id={client_id} "
                f"alpha={alpha} seed={split_seed} local_size={len(client_idx)}"
            )
        else:
            # IID 균등 등분
            g = torch.Generator(); g.manual_seed(split_seed)
            N = len(full_dataset)
            perm = torch.randperm(N, generator=g).tolist()
            q, r = divmod(N, num_clients)
            sizes = [q + (1 if i < r else 0) for i in range(num_clients)]
            start = sum(sizes[:client_id]); end = start + sizes[client_id]
            client_idx = perm[start:end]
            base_dataset = Subset(full_dataset, client_idx)
            logging.info(
                f"[IID] clients={num_clients} id={client_id} seed={split_seed} "
                f"local_size={len(client_idx)}"
            )

    # —— train / val / test 분할 (기존 로직 유지, 대상만 base_dataset로 변경) ——
    test_split = TEST_SPLIT_FIXED
    val_split  = float(validation_split) if validation_split is not None else VAL_SPLIT_DEFAULT
    assert 0.0 < val_split < 1.0, "validation_split must be (0,1)"
    assert 0.0 < test_split < 1.0 and val_split + test_split < 1.0, \
        "validation_split + test_split must be < 1"

    base_len = len(base_dataset)
    train_size = int((1 - val_split - test_split) * base_len)
    validation_size = int(val_split * base_len)
    test_size = base_len - train_size - validation_size

    g2 = torch.Generator(); g2.manual_seed(split_seed + client_id)
    train_dataset, val_dataset, test_dataset = random_split(
        base_dataset, [train_size, validation_size, test_size], generator=g2
    )

    # DataLoader 반환 (기존 유지)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size)
    return train_loader, val_loader, test_loader


def gl_model_torch_validation(batch_size):
    """
    Setting up a dataset to evaluate a global model on the server (기존 그대로)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    val_dataset = datasets.MNIST(root='./dataset/mnist', train=False, download=True, transform=transform)
    gl_val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return gl_val_loader
