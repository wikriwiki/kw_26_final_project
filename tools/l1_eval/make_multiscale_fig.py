"""단위별 경계 — 차원마다 굵은 단위 → 세밀한 단위 사다리 그림."""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 9.5,
    'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#ecebe7',
    'grid.linewidth': 0.8, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
})
SIM, UNI, INF, INK = '#2a78d6', '#1baf7a', '#eb6834', '#0b0b0b'


def tier(r, presence=False):
    lo, hi = r['S_sim_ci']
    if r['S_sim'] >= r.get('S_ceil_band', [r['S_ceil'], 0])[0] - 1e-9 and r['S_sim'] >= r['S_ceil'] - 0.002:
        return '●'
    if presence:
        return '◐' if lo > r['S_base'] else ('○' if hi < r['S_base'] else '△')
    b = r['S_base_all']
    uni = b.get('균등')
    inf = max([v for k, v in b.items() if k != '균등'] or [-1])
    if lo > max(uni, inf):
        return '◐'
    if lo > uni:
        return '◔'
    if hi < uni:
        return '○'
    return '△'


def main(res):
    R = json.load(open(f'{res}/l1_multiscale.json', encoding='utf-8'))
    dims = list(R['pooled'])
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.6))
    axes = axs.ravel()
    for ax, dim in zip(axes[:5], dims + ['사람 위치']):
        rows = R['pooled'][dim] if dim != '사람 위치' else R['presence']
        x = np.arange(len(rows))
        sim = [r['S_sim'] for r in rows]
        lo = [r['S_sim_ci'][0] for r in rows]
        hi = [r['S_sim_ci'][1] for r in rows]
        ceil = [r['S_ceil'] for r in rows]
        ax.plot(x, ceil, color=INK, linestyle=(0, (3, 3)), linewidth=1.3, label='완벽한 모형(같은 표본 크기)')
        if dim != '사람 위치':
            uni = [r['S_base_all'].get('균등') for r in rows]
            inf = [max([v for k, v in r['S_base_all'].items() if k != '균등'] or [np.nan]) for r in rows]
            ax.plot(x, inf, color=INF, linewidth=2, marker='s', markersize=5.5, markeredgecolor='#fcfcfb', label='실제 자료 기반 단순 모형')
            ax.plot(x, uni, color=UNI, linewidth=2, marker='D', markersize=5, markeredgecolor='#fcfcfb', label='균등(정보 없음)')
        else:
            ax.plot(x, [r['S_base'] for r in rows], color=INF, linewidth=2, marker='s', markersize=5.5, markeredgecolor='#fcfcfb')
        ax.fill_between(x, lo, hi, color=SIM, alpha=0.15, linewidth=0)
        ax.plot(x, sim, color=SIM, linewidth=2.4, marker='o', markersize=7, markeredgecolor='#fcfcfb', markeredgewidth=1.4, label='시뮬레이션')
        for xi, r in zip(x, rows):
            t = tier(r, dim == '사람 위치')
            word = {'●': '재현', '◐': '부가가치', '◔': '정보 있음', '△': '균등 수준', '○': '왜곡'}[t]
            col = {'○': '#a32638', '△': '#8a6210'}.get(t, SIM)
            ax.text(xi, r['S_sim'] - 0.03, word, ha='center', va='top', fontsize=8.5, color=col, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([r['scale'].replace('×', '×\n').replace(' ', '\n', 1) if len(r['scale']) > 6 else r['scale'] for r in rows], fontsize=8.5)
        ax.set_title(dim + (' (평일 14시)' if dim == '사람 위치' else ' (지출)'), loc='left', fontsize=11.5, fontweight='bold', color=INK, pad=8)
        ax.set_ylim(0.5, 1.02)
        ax.set_axisbelow(True)
        ax.grid(axis='x', visible=False)
        ax.set_xlabel('굵은 단위 → 세밀한 단위', fontsize=9)
        ax.set_ylabel('실측과의 가까움 (1 - JSD)')
    h, l = axes[0].get_legend_handles_labels()
    lg = axes[5]
    lg.axis('off')
    lg.legend(h, l, loc='upper left', frameon=False, fontsize=10.5, bbox_to_anchor=(0.02, 1.0))
    expl = ['판정 (점 아래 글자)', '재현: 완벽한 모형 수준', '부가가치: 실제 자료 기반 단순 모형보다 나음',
            '정보 있음: 균등보다 나음 (실제 패턴 일부 담음)', '균등 수준: 균등과 구분 안 됨', '왜곡: 균등보다 못함',
            '', '사람 위치의 단순 모형 = 하루 종일 집에 있음']
    for i, t_ in enumerate(expl):
        lg.text(0.04, 0.52 - i * 0.068, t_, transform=lg.transAxes, fontsize=10 if i else 10.5,
                fontweight='bold' if i == 0 else 'normal', color=INK if i in (0, 7) else '#52514e')
    fig.tight_layout()
    fig.savefig(f'{res}/figM_ladders.png', dpi=170)
    plt.close(fig)
    print('ok')


if __name__ == '__main__':
    main(sys.argv[1])
