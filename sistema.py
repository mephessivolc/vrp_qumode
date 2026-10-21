"""
sistema.py — memória e CPU disponíveis no servidor, SEM sudo e sem comandos de
terminal. Colocar na raiz do projeto.

Por que não basta olhar a memória total da máquina
--------------------------------------------------
Em servidor compartilhado ou com gerenciador de tarefas (SLURM, contêiner), o
limite real do seu processo pode ser MENOR que a memória física. Por isso lemos
três fontes, todas legíveis sem privilégio, e usamos a MENOR:

    /proc/meminfo                       MemAvailable: o que cabe sem trocar para o disco
    cgroup v2 / v1                      limite do seu contêiner ou tarefa e o uso atual
    variáveis SLURM_MEM_PER_NODE / ...  memória alocada ao job (se existir)

MemAvailable já desconta o que outros usuários estão usando NO MOMENTO da leitura;
ela pode mudar depois, então a memória máxima automática usa só uma fração
(padrão 70%).
"""
import os
import platform
from typing import Dict, Optional


def _ler(caminho: str) -> Optional[str]:
    try:
        with open(caminho, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return None


def _meminfo_kb() -> Dict[str, int]:
    txt = _ler("/proc/meminfo")
    out: Dict[str, int] = {}
    if not txt:
        return out
    for linha in txt.splitlines():
        if ":" not in linha:
            continue
        chave, valor = linha.split(":", 1)
        partes = valor.split()
        if partes and partes[0].isdigit():
            out[chave.strip()] = int(partes[0])          # kB
    return out


def _cgroup():
    """(limite_bytes, uso_bytes) do cgroup v2 ou v1; (None, None) sem limite."""
    lim = _ler("/sys/fs/cgroup/memory.max")              # cgroup v2
    if lim is not None:
        uso = _ler("/sys/fs/cgroup/memory.current")
        if lim.isdigit():
            return int(lim), (int(uso) if uso and uso.isdigit() else None)
        return None, None                                # "max": sem limite
    lim = _ler("/sys/fs/cgroup/memory/memory.limit_in_bytes")   # cgroup v1
    if lim and lim.isdigit() and int(lim) < (1 << 60):
        uso = _ler("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        return int(lim), (int(uso) if uso and uso.isdigit() else None)
    return None, None


def memoria_gb() -> Dict[str, Optional[float]]:
    """Dicionário com total, disponível, limite do cgroup, alocação SLURM e
    'efetivo_gb' (a MENOR memória realmente utilizável)."""
    gb = 1024.0 ** 3
    info: Dict[str, Optional[float]] = {}
    mi = _meminfo_kb()
    if mi:
        info["total_gb"] = mi.get("MemTotal", 0) / 1024.0 ** 2
        info["disponivel_gb"] = mi.get("MemAvailable", mi.get("MemFree", 0)) / 1024.0 ** 2
        info["swap_total_gb"] = mi.get("SwapTotal", 0) / 1024.0 ** 2
    else:                                                # sem /proc (macOS, Windows)
        try:
            import psutil
            vm = psutil.virtual_memory()
            info["total_gb"] = vm.total / gb
            info["disponivel_gb"] = vm.available / gb
        except ImportError:
            pass
    lim, uso = _cgroup()
    if lim is not None:
        info["cgroup_limite_gb"] = lim / gb
        if uso is not None:
            info["cgroup_livre_gb"] = (lim - uso) / gb
    por_no = os.environ.get("SLURM_MEM_PER_NODE")
    por_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    cpus = os.environ.get("SLURM_CPUS_ON_NODE")
    if por_no and por_no.isdigit():
        info["slurm_gb"] = int(por_no) / 1024.0
    elif por_cpu and por_cpu.isdigit() and cpus and cpus.isdigit():
        info["slurm_gb"] = int(por_cpu) * int(cpus) / 1024.0
    candidatos = [info[k] for k in ("disponivel_gb", "cgroup_livre_gb", "slurm_gb")
                  if info.get(k) is not None]
    info["efetivo_gb"] = min(candidatos) if candidatos else None
    return info


def cpus_disponiveis() -> Optional[int]:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def memoria_max_automatica(fracao: float = 0.7, padrao_gb: float = 8.0) -> float:
    """Memória máxima a usar (GB) = fracao * memória efetiva. Se não for possível
    detectar a memória, devolve padrao_gb (conservador)."""
    ef = memoria_gb().get("efetivo_gb")
    return float(fracao * ef) if ef else float(padrao_gb)


def relatorio_sistema(fracao: float = 0.7) -> str:
    m = memoria_gb()

    def f(chave, rotulo):
        v = m.get(chave)
        return f"  {rotulo:<44}{'—' if v is None else f'{v:10.2f} GB'}"

    linhas = [
        "=" * 70,
        "SISTEMA (leitura sem privilégio; /proc, cgroups e variáveis do SLURM)",
        "=" * 70,
        f"  máquina: {platform.node()}  |  {platform.system()} {platform.release()}",
        f"  CPUs utilizáveis por este processo: {cpus_disponiveis()}",
        f("total_gb", "memória física total"),
        f("disponivel_gb", "disponível agora (MemAvailable)"),
        f("swap_total_gb", "swap total"),
        f("cgroup_limite_gb", "limite do contêiner/tarefa (cgroup)"),
        f("cgroup_livre_gb", "livre dentro do limite do cgroup"),
        f("slurm_gb", "alocada pelo SLURM ao job"),
        f("efetivo_gb", "EFETIVA (a menor das disponíveis)"),
    ]
    ef = m.get("efetivo_gb")
    if ef:
        linhas.append(f"  memória máxima automática ({fracao:.0%} da efetiva): {fracao * ef:.2f} GB")
    else:
        linhas.append("  não foi possível detectar a memória: o padrão conservador é 8 GB")
    linhas.append("  Atenção: em servidor compartilhado, a memória disponível pode mudar "
                  "durante a execução.")
    return "\n".join(linhas)