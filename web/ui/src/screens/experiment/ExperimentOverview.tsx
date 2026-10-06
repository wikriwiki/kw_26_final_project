/**
 * 실험 개요 — 팔마다 어디까지 왔는지, 데이터를 언제 받았는지.
 *
 * 진행 막대의 분모는 실험 정의의 날짜 수다. 받은 날 중 대상자 전원이 채워진 날만 "완료"로 센다.
 * 봉인 검사에 실패한 줄이 있으면 숨기지 않고 경고로 올린다.
 */
import { useEffect, useRef, useState } from 'react';
import { Badge } from '../../components/Badge';
import { Button } from '../../components/Button';
import { Card, Stat } from '../../components/Card';
import { Callout } from '../../components/Feedback';
import { Meter } from '../../components/Meter';
import { dateTime, int } from '../../lib/format';
import { experimentsApi, PHASE_TEXT, ROLE_LABEL } from '../../lib/experiments';
import type { ArmStatus, SyncStatus } from '../../lib/experiments';
import { READ_ONLY } from '../../lib/runtime';
import { useExperiment } from './ExperimentWorkspace';

function ArmCard({ arm, cohort }: { arm: ArmStatus; cohort: number }) {
  const current = arm.current_day;
  const tone = arm.phase === 'complete' ? 'ok' : arm.phase === 'running' ? 'info' : 'neutral';
  return (
    <Card
      title={arm.label}
      aside={<Badge tone={tone}>{PHASE_TEXT[arm.phase]}</Badge>}
      note={`${ROLE_LABEL[arm.role]} · ${arm.start}부터 ${int(arm.days)}일${arm.parent ? ' · 공통 사전에서 갈라짐' : ''}`}
      className="c4"
    >
      <div className="armcard__progress">
        <Meter ratio={arm.days ? arm.days_complete / arm.days : 0} label={`${arm.label} 완료 일수`} />
        <span className="num">
          {int(arm.days_complete)} / {int(arm.days)}일
        </span>
      </div>
      <dl className="armcard__facts">
        <div>
          <dt>진행 중인 날</dt>
          <dd className="num">{current ? `${current.day} · ${int(current.rows ?? 0)} / ${int(cohort)}명` : '없음'}</dd>
        </div>
        <div>
          <dt>건너뛴 대상자-일</dt>
          <dd className="num">{int(arm.skipped_total)}</dd>
        </div>
        <div>
          <dt>모델</dt>
          <dd>{arm.manifest?.model ?? '기록 없음'}</dd>
        </div>
        <div>
          <dt>마지막으로 받은 시각</dt>
          <dd className="num">{arm.synced_at ? dateTime(arm.synced_at) : '받은 적 없음'}</dd>
        </div>
      </dl>
      {arm.integrity_failures > 0 ? (
        <Callout tone="warn">봉인 검사에 실패한 줄 {int(arm.integrity_failures)}개를 집계에서 뺐습니다.</Callout>
      ) : null}
    </Card>
  );
}

function SyncCard({ experimentId, onDone }: { experimentId: string; onDone: () => void }) {
  const [status, setStatus] = useState<SyncStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const wasRunning = useRef(false);

  useEffect(() => {
    let alive = true;
    let timer: number | undefined;
    const poll = () =>
      experimentsApi.syncStatus(experimentId).then(
        (next) => {
          if (!alive) return;
          // 받는 중이던 것이 끝나면 화면의 진행 정보를 새로 읽는다
          if (wasRunning.current && next.state === 'done') onDone();
          wasRunning.current = next.state === 'running';
          setStatus(next);
          if (next.state === 'running') timer = window.setTimeout(poll, 2000);
        },
        (err) => alive && setError(err.message),
      );
    poll();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [experimentId, onDone, status?.started_at]);

  const running = status?.state === 'running';
  const start = () => {
    setError(null);
    experimentsApi.syncStart(experimentId).then(setStatus, (err) => setError(err.message));
  };

  return (
    <Card title="실행 기록 받기" note="실험 서버에서 읽기만 합니다. 서버의 파일·실행에는 손대지 않습니다." className="c12">
      <div className="synccard">
        <p className="synccard__state">
          {!status
            ? '상태 확인 중…'
            : !status.remote_configured
              ? `이 PC 에 원격 접속이 설정되지 않았습니다 (SIM_REMOTE_${status.remote_profile ?? '…'}). 이미 받아 둔 기록만 보여 줍니다.`
              : running
                ? `받는 중 — ${status.message ?? ''}`
                : status.state === 'failed'
                  ? `실패: ${status.error ?? '알 수 없는 오류'}`
                  : status.state === 'done'
                    ? `완료 ${status.finished_at ? dateTime(status.finished_at) : ''}`
                    : '지금 받으면 새로 쌓인 날과 진행 중인 날의 늘어난 부분만 받습니다.'}
        </p>
        {!READ_ONLY ? (
          <Button
            variant="secondary"
            onClick={start}
            disabled={!status?.remote_configured || running}
            busy={running}
            busyLabel="받는 중"
          >
            지금 받기
          </Button>
        ) : null}
      </div>
      {error ? <Callout tone="warn">{error}</Callout> : null}
    </Card>
  );
}

export function ExperimentOverview() {
  const { experiment, reload } = useExperiment();
  const cohort = experiment.cohort_size;
  const totalSkipped = experiment.arms.reduce((sum, arm) => sum + arm.skipped_total, 0);
  const observedDays = experiment.arms.reduce((sum, arm) => sum + arm.days_present, 0);

  return (
    <div className="stack">
      {experiment.interpretation ? <Callout tone="info">{experiment.interpretation}</Callout> : null}
      <div className="grid">
        <Stat className="c3" label="대상자" value={int(cohort)} unit="명" />
        <Stat className="c3" label="기간" value={experiment.window.start ? `${experiment.window.start}부터 ${int(experiment.window.days)}일` : null} />
        <Stat className="c3" label="정책 시행일" value={experiment.window.effective_date ?? null} />
        <Stat className="c3" label="받은 팔-일 / 건너뛴 대상자-일" value={`${int(observedDays)} / ${int(totalSkipped)}`} />
      </div>
      <div className="grid">
        {experiment.arms.map((arm) => (
          <ArmCard key={arm.arm_id} arm={arm} cohort={cohort} />
        ))}
      </div>
      <SyncCard experimentId={experiment.experiment_id} onDone={reload} />
      {experiment.analysis.limitations?.length ? (
        <Card title="읽을 때 지킬 것" className="c12">
          <ul className="plainlist">
            {experiment.analysis.limitations.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </Card>
      ) : null}
    </div>
  );
}
