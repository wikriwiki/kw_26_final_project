/**
 * 시각화 — 스펙 §7 "시각화 페이지 진입 흐름".
 * 단일 목적: 지도 위에서 시뮬레이션을 관찰한다.
 *
 * **지도가 주인공이다.** 지도는 전체 폭·전체 높이를 쓰고, 그 위에 얹는 것은 세 가지뿐이다.
 *   1. 좌상단 "← 결과로" — 어디서 왔는지, 어떻게 돌아가는지 (§9 back-behavior)
 *   2. 하단 일자 선택 — 주소에 그대로 반영돼 링크로 공유된다 (§9 deep-linking)
 *   3. 우측 표시 설정 — **기본 닫힘** (§8 progressive-disclosure)
 *
 * 재생·속도 조절은 여기에 두지 않는다. 그건 지도 자체의 조작이라 임베드한 뷰어가 갖고 있고,
 * 껍데기가 같은 버튼을 한 벌 더 그리면 어느 쪽이 진짜인지 알 수 없게 된다.
 *
 * **run 은 셸이 소유한다** (설계도 §5). 상단바에서 run 라벨을, 표시 설정에서 run 선택기를
 * 걷어냈다. 상단바에 남는 것은 "결과로" 복귀와 지도가 무엇을 담고 있는지뿐이다.
 */
import { useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useRun } from '../app/RunContext';
import type { RunContextValue } from '../app/RunContext';
import { Badge } from '../components/Badge';
import { Button } from '../components/Button';
import { SelectField } from '../components/Field';
import { ArrowLeftIcon, ChevronLeftIcon, ChevronRightIcon, LayersIcon, XIcon } from '../components/Icon';
import { useDayParam } from '../app/useViewState';
import { api } from '../lib/api';
import { int } from '../lib/format';

/**
 * 지도는 데이터 팩(`scripts/sim/visualization_3d/pack.py`)을 껍데기(`/viz/shell.html`)에
 * 물려 그린다. 껍데기는 수 KB 이고, 지도를 처음 그리는 데 받는 데이터는 시연 표본 기준
 * 약 1.2MB 다(예전 단일 HTML 은 133MB). 사람을 고를 때만 그 사람의 기억·일정을 받는다.
 *
 * 팩이 하나도 없거나 API 가 없으면 예전 단일 HTML 산출물로 돌아간다.
 */
const LEGACY_EMBED_SRC = '/viz/standalone.html';
const LEGACY_EMBED_CASE = { runId: 'SEOUL7500', label: '이 실행의 활동 기록' } as const;

interface VizEmbed {
  src: string;
  /** 지도가 실제로 담고 있는 실행(또는 팩) 이름 */
  caseId: string;
  label: string;
  /** 지금 고른 실행의 지도인가 */
  exact: boolean;
}

function legacyEmbed(runId: string): VizEmbed {
  return {
    src: LEGACY_EMBED_SRC,
    caseId: LEGACY_EMBED_CASE.runId,
    label: LEGACY_EMBED_CASE.label,
    exact: runId === LEGACY_EMBED_CASE.runId,
  };
}

/**
 * 고른 실행에 맞는 팩을 찾는다. 같은 실행의 팩이 없으면 첫 팩(표본)을 띄우되,
 * 그 사실을 화면에 적는다 — 다른 실행의 지도를 이 실행의 것처럼 보이지 않게 한다.
 */
function useVizEmbed(runId: string): { embed: VizEmbed | null; loading: boolean } {
  const [state, setState] = useState<{ embed: VizEmbed | null; loading: boolean }>({ embed: null, loading: true });
  useEffect(() => {
    let alive = true;
    api.listVizPacks().then(
      (packs) => {
        if (!alive) return;
        const usable = packs.items.filter((item) => !item.error);
        const exact = usable.find((item) => item.run_id === runId || item.pack_id === runId);
        const chosen = exact ?? usable[0];
        if (!chosen) {
          setState({ embed: legacyEmbed(runId), loading: false });
          return;
        }
        const packPath = `/viz/packs/${encodeURIComponent(chosen.pack_id)}/`;
        setState({
          embed: {
            src: `/viz/shell.html?theme=console&pack=${encodeURIComponent(packPath)}`,
            caseId: chosen.run_id ?? chosen.pack_id,
            label: chosen.label ?? LEGACY_EMBED_CASE.label,
            exact: Boolean(exact),
          },
          loading: false,
        });
      },
      () => {
        if (alive) setState({ embed: legacyEmbed(runId), loading: false });
      },
    );
    return () => {
      alive = false;
    };
  }, [runId]);
  return state;
}

/** `scripts/sim/visualization_3d/template.html` 의 범례·필터와 같은 항목을 쓴다 */
const COLOR_MODES = [
  { value: 'dist', label: '자치구별' },
  { value: 'cat', label: '활동 종류별' },
  { value: 'appointment', label: '약속 있는 대상자' },
];

const HEAT_MODES = [
  { value: 'off', label: '표시 안 함' },
  { value: 'density', label: '대상자 수' },
  { value: 'spending', label: '소비 금액' },
];

const DISTRICTS = [
  '전체', '종로구', '중구', '용산구', '성동구', '광진구', '동대문구', '중랑구', '성북구',
  '강북구', '도봉구', '노원구', '은평구', '서대문구', '마포구', '양천구', '강서구', '구로구',
  '금천구', '영등포구', '동작구', '관악구', '서초구', '강남구', '송파구', '강동구',
];

const LAYERS: Array<{ key: string; label: string }> = [
  { key: 'trails', label: '이동 잔상' },
  { key: 'heatmap', label: '히트맵' },
  { key: 'boundary', label: '구 경계' },
  { key: 'policyZones', label: '정책 구역' },
];

export function VisualizationScreen() {
  const run = useRun();
  // 실행이 바뀌면 패널·레이어 상태를 그 실행의 기본값에서 다시 시작한다
  return <VisualizationView key={run.id} run={run} />;
}

function VisualizationView({ run }: { run: RunContextValue }) {
  const runId = run.id;
  const navigate = useNavigate();
  const location = useLocation();
  const [day, days, setDay] = useDayParam(run.bundle);
  const [panelOpen, setPanelOpen] = useState(false);
  const [layers, setLayers] = useState<Record<string, boolean>>({
    trails: true,
    heatmap: false,
    boundary: false,
    policyZones: true,
  });
  const panelBtnRef = useRef<HTMLButtonElement>(null);
  const panelCloseRef = useRef<HTMLButtonElement>(null);

  const fromResults = (location.state as { from?: string } | null)?.from === 'results';
  const dayIndex = Math.max(0, days.indexOf(day));

  useEffect(() => {
    if (panelOpen) panelCloseRef.current?.focus();
  }, [panelOpen]);

  function closePanel() {
    panelBtnRef.current?.focus();
    setPanelOpen(false);
  }

  /** 결과에서 들어왔으면 진짜 뒤로 간다 — 그래야 보던 스크롤 위치가 살아난다 */
  function goToResults() {
    if (fromResults) {
      navigate(-1);
      return;
    }
    navigate(run.path('results'));
  }

  /**
   * src 에 day 를 붙이지 않는다. 붙이면 일자를 옮길 때마다 URL 이 바뀌어
   * iframe 이 지도를 처음부터 다시 연다. 팩은 자체 타임라인을 들고 있으므로
   * 콘솔이 일자를 지시할 필요도 없다.
   */
  const { embed, loading: embedLoading } = useVizEmbed(runId);
  const embedSrc = embed?.src ?? null;
  // 팩 목록을 받는 동안에도 지도 자리를 지킨다 — 빈 자리 안내가 잠깐 비쳤다 사라지지 않게
  const embedMode = embedLoading || embedSrc !== null;
  const caseMismatch = embed !== null && !embed.exact;

  return (
    <div className="viz">
      <header className="viz__bar">
        <button type="button" className="viz__back" onClick={goToResults}>
          <ArrowLeftIcon size={18} />
          <span>결과로</span>
        </button>

        {/*
          실행 이름·상태는 사이드바가 항상 말한다. 여기서 한 번 더 적으면 같은 사실이
          두 곳에 있게 되고, 둘이 어긋나는 순간 어느 쪽이 맞는지 알 수 없다 (IA §5).
          그래서 제목은 **이 화면이 무엇인지**만 말하고, 아래 줄은 **지도가 무엇을 담고
          있는지**만 말한다.
        */}
        <div className="viz__id">
          <h1 className="viz__title">3D 지도</h1>
          <p className="viz__meta">
            {embedMode ? (
              /* 지도가 자체 타임라인을 들고 있다. 콘솔이 날짜를 따로 적으면
                 지도 HUD 의 날짜와 어긋나 어느 쪽이 맞는지 알 수 없게 된다.
                 기간은 지도가 말하게 두고, 여기서는 무엇을 담은 지도인지만 밝힌다. */
              <span className="viz__meta-extra">
                {embed ? `${embed.label} · 기간은 지도 안에 표시됩니다` : '지도 데이터를 찾는 중'}
                {caseMismatch ? (
                  /* 고른 실행의 지도가 없어 표본을 띄웠다는 사실은 지도를 가리지 않게
                     이 줄에서 말한다. 예전에는 지도 위 상자가 검색창을 덮었다 */
                  <span className="viz__sample">
                    <Badge tone="warn">표본 지도</Badge>
                    {` 실행 ${runId} 의 지도가 아직 없어 ${embed?.caseId} 을(를) 보여 줍니다`}
                  </span>
                ) : null}
              </span>
            ) : (
              <>
                <span className="num">{day}</span>
                {days.length > 1 ? (
                  <span className="viz__meta-extra">{` · ${int(dayIndex + 1)}일째 / ${int(days.length)}일`}</span>
                ) : null}
              </>
            )}
          </p>
        </div>

        {/*
          지도가 붙어 있을 때는 숨긴다. 임베드된 뷰어가 색상 모드·히트맵·
          자치구 필터를 자기 범례 패널에 이미 갖고 있어서, 같은 항목을 조작하는
          패널이 둘이 되면 서로 반영되지 않는 설정이 두 벌 생긴다.
        */}
        {!embedMode && (
          <Button
            ref={panelBtnRef}
            variant="secondary"
            icon={<LayersIcon size={18} />}
            aria-expanded={panelOpen}
            aria-controls="viz-panel"
            onClick={() => (panelOpen ? closePanel() : setPanelOpen(true))}
          >
            표시 설정
          </Button>
        )}
      </header>

      <div className="viz__stage">
        {/* 지도 자리 — 실제 뷰어가 들어올 컨테이너 */}
        <div className="mapstage">
          {embedLoading ? (
            <p className="mapstage__notice" role="status">
              지도 데이터를 찾는 중입니다…
            </p>
          ) : embed && embedSrc ? (
            <>
              <iframe className="mapstage__frame" src={embedSrc} title={`${embed.caseId} 3D 지도`} />
            </>
          ) : (
            <div className="mapstage__placeholder">
              <p className="mapstage__title">지도를 여기에 붙입니다</p>
              <p className="mapstage__body">
                실행 {runId} 의 {day} 3D 지도가 이 영역을 가득 채웁니다. 지금은 화면 구성과 조작
                흐름만 확인할 수 있습니다.
              </p>
            </div>
          )}
        </div>

        {/*
          일자 선택 — 주소의 day 와 같은 값이라 링크를 그대로 공유할 수 있다.

          지도가 붙어 있을 때는 숨긴다. 임베드된 뷰어가 자체 재생 컨트롤과
          타임라인을 이미 갖고 있어서, 같은 화면에 일자 조작기가 둘이 되면
          둘 중 어느 것이 진짜인지 알 수 없고 서로 동기화되지도 않는다.
          조작 주체는 하나여야 한다.
        */}
        {!embedMode && (
        <div className="viztransport">
          <span className="viztransport__label" id="viz-day-label">
            일자
          </span>
          <button
            type="button"
            className="viztransport__step"
            onClick={() => setDay(days[dayIndex - 1] ?? day)}
            disabled={dayIndex <= 0}
            aria-label="이전 일자"
          >
            <ChevronLeftIcon size={18} />
          </button>
          <span className="viztransport__day num">{day}</span>
          <button
            type="button"
            className="viztransport__step"
            onClick={() => setDay(days[dayIndex + 1] ?? day)}
            disabled={dayIndex >= days.length - 1}
            aria-label="다음 일자"
          >
            <ChevronRightIcon size={18} />
          </button>
          <input
            className="viztransport__range"
            type="range"
            min={0}
            max={Math.max(0, days.length - 1)}
            step={1}
            value={dayIndex}
            onChange={(e) => setDay(days[Number(e.target.value)] ?? day)}
            aria-labelledby="viz-day-label"
            aria-valuetext={day}
          />
        </div>
        )}

        {/* 표시 설정 — 기본 닫힘 */}
        {panelOpen ? (
          <aside className="vizpanel" id="viz-panel" aria-label="표시 설정">
            <div className="vizpanel__head">
              <h2 className="vizpanel__title">표시 설정</h2>
              <button
                ref={panelCloseRef}
                type="button"
                className="vizpanel__close"
                onClick={closePanel}
                aria-label="표시 설정 닫기"
              >
                <XIcon size={18} />
              </button>
            </div>

            <div className="vizpanel__body">
              {/* 실행 선택기는 여기 없다 — 셸이 소유한다 (IA §5) */}
              <fieldset className="vizpanel__group">
                <legend className="vizpanel__legend">겹쳐 볼 것</legend>
                {LAYERS.map(({ key, label }) => (
                  <label className="vizpanel__opt" key={key}>
                    <input
                      type="checkbox"
                      checked={layers[key] ?? false}
                      onChange={(e) =>
                        setLayers((prev) => ({ ...prev, [key]: e.target.checked }))
                      }
                    />
                    <span>{label}</span>
                  </label>
                ))}
              </fieldset>

              <SelectField label="점 색 기준" defaultValue="dist" options={COLOR_MODES} />
              <SelectField label="히트맵 기준" defaultValue="off" options={HEAT_MODES} />
              <SelectField
                label="자치구"
                defaultValue="전체"
                options={DISTRICTS.map((d) => ({ value: d, label: d }))}
              />

              <p className="vizpanel__note">
                표시 설정은 지도를 연결한 뒤에 적용됩니다. 지금 바꿔도 화면은 달라지지 않습니다.
              </p>
            </div>
          </aside>
        ) : null}
      </div>
    </div>
  );
}
