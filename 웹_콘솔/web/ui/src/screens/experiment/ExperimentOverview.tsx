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
import { dateTime, dayRange, int, lastDay } from '../../lib/format';
import { experimentsApi, PHASE_TEXT } from '../../lib/experiments';
import type { ArmStatus, SyncStatus } from '../../lib/experiments';
import { CAN_EDIT } from '../../lib/runtime';
import { useExperiment } from './ExperimentWorkspace';

function ArmCard({ arm, cohort }: { arm: ArmStatus; cohort: number }) {
  const current = arm.current_day;
  const tone = arm.phase === 'complete' ? 'ok' : arm.phase === 'running' ? 'info' : 'neutral';
  return (
    <Card
      title={arm.label}
      aside={<Badge tone={tone}>{PHASE_TEXT[arm.phase]}</Badge>}
      note={dayRange(arm.start, arm.days)}
      className="c4"
    >
      {/* 다 끝난 쪽은 기간만으로 충분하다. 진행 정보는 아직 도는 쪽에만 보인다 */}
      {arm.phase === 'complete' ? null : (
        <div className="armcard__progress">
          <Meter ratio={arm.days ? arm.days_complete / arm.days : 0} label={`${arm.label} 완료 일수`} />
          <span className="num">
            {int(arm.days_complete)} / {int(arm.days)}일
          </span>
        </div>
      )}
      {current && arm.phase !== 'complete' ? (
        <p className="card__note num">
          진행 중 {current.day} · {int(current.rows ?? 0)} / {int(cohort)}명
        </p>
      ) : null}
      {arm.integrity_failures > 0 ? (
        <Callout tone="warn">검사에 실패한 기록 {int(arm.integrity_failures)}건은 집계에서 뺐습니다.</Callout>
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
  // 이 PC 에 원격 접속이 없으면 받을 수 없다. 이미 받아 둔 기록으로 보는 화면이라 카드를 그리지 않는다
  if (status && !status.remote_configured && !error) return null;
  const start = () => {
    setError(null);
    experimentsApi.syncStart(experimentId).then(setStatus, (err) => setError(err.message));
  };

  return (
    <Card title="실행 기록 받기" note="실험 서버에서 읽기만 합니다." className="c12">
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
                    : '새로 쌓인 부분만 받습니다.'}
        </p>
        {CAN_EDIT ? (
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
  const { start, days } = experiment.window;
  const pre = experiment.arms.find((arm) => arm.role === 'shared_pre');
  const post = experiment.arms.find((arm) => arm.role === 'treatment');

  return (
    <div className="stack">
      {experiment.interpretation ? <Callout tone="info">{experiment.interpretation}</Callout> : null}
      <div className="grid">
        <Stat className="c3" label="시민" value={int(cohort)} unit="명" />
        {/* 큰 숫자 칸에는 월-일만 둔다. 해와 일수는 아래 줄로 — 날짜가 중간에서 꺾이지 않게 */}
        <Stat
          className="c3"
          label="기간"
          value={start && days ? `${start.slice(5)} ~ ${lastDay(start, days).slice(5)}` : null}
          hint={start && days ? `${start.slice(0, 4)}년 · ${int(days)}일` : undefined}
        />
        <Stat className="c3" label="정책 시행일" value={experiment.window.effective_date ?? null} />
        <Stat
          className="c3"
          label="비교"
          value={pre && post ? `${int(pre.days)}일 + ${int(post.days)}일` : null}
          hint="정책 전 함께 → 있음·없음으로 나눔"
          unknownReason="—"
        />
      </div>
      <div className="grid">
        {experiment.arms.map((arm) => (
          <ArmCard key={arm.arm_id} arm={arm} cohort={cohort} />
        ))}
      </div>
      {/* 원격 실험 서버에서 받아 오는 실험만 "받기" 카드가 뜻이 있다. 결과 파일에서 들여온 실험에는 그리지 않는다 */}
      {experiment.remote_profile ? <SyncCard experimentId={experiment.experiment_id} onDone={reload} /> : null}
      {experiment.analysis.limitations?.length ? (
        <Card title="참고" className="c12">
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
