"""연결 실패는 그 칸의 결과가 아니라 서버가 사라졌다는 뜻이다.

2026-09-21: xgrammar abort 가 서버를 죽였고, 계획기는 그것을 모른 채 950칸 중
694칸을 `eligible=False` 로 기록했다. 빈 값이면 비교에서 빠지지만 틀린 값은
"이 프롬프트가 자원 게이트를 자주 못 넘는다"로 읽힌다. 라운드 하나를 버렸다.

그래서 연결 실패만 따로 세고, 연속으로 쌓이면 라운드를 멈춘다. 여기서 고정하는
것은 **무엇을 기반 고장으로 볼지**다 — 너무 넓으면 멀쩡한 계약 위반에도 멈추고,
너무 좁으면 또 놓친다.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from validate_action_planner import INFRA, INFRA_RE, ServerGone


def test_connection_failures_are_recognised():
    for message in ('<urlopen error [Errno 111] Connection refused>',
                    'Connection reset by peer',
                    'Remote end closed connection without response',
                    'Max retries exceeded with url: /generate'):
        assert INFRA_RE.search(message), message


def test_ordinary_contract_failures_are_not_infrastructure():
    """계약 위반은 그 칸의 진짜 결과다. 이걸로 멈추면 안 된다."""
    for message in ('incomplete_generation',
                    'Commitment lacks evidence',
                    'Purchase quotes differ from preview supplied before planning',
                    'Simultaneous committed starts unsupported'):
        assert not INFRA_RE.search(message), message


def test_a_hung_server_counts_too():
    """13:21 의 멈춤은 timed out 으로, 15:33 의 죽음은 Connection refused 로 나타났다."""
    assert INFRA_RE.search('timed out')


def test_one_slow_cell_cannot_stop_the_round():
    """연속이라는 조건이 느린 칸 하나를 걸러낸다 — 답이 오면 계수기가 0이 된다."""
    assert INFRA['limit'] >= 8
    src = (Path(__file__).resolve().parents[3]
           / 'scripts/sim/validate_action_planner.py').read_text(encoding='utf-8')
    assert "INFRA['count'] = 0     # 한 번이라도 답이 오면" in src


def test_server_gone_is_an_error_not_a_row():
    assert issubclass(ServerGone, RuntimeError)
