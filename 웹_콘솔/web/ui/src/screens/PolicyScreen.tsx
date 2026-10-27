/**
 * 이 실행에 적용된 정책 — 설계도 §5, 라우트 `/runs/:runId/policy`.
 *
 * **화면의 성격이 바뀌었다.** 전에는 "정책을 골라 검증하는" 편집 화면이었지만,
 * 정책을 고르고 검증하는 일은 `/new`(새 시뮬레이션 만들기)로 옮겼다 — 정책이 실제로 쓰이는 시점이다.
 * 여기는 **이미 끝난(또는 도는) 실행에 무엇이 적용됐는지 확인하는 읽기 화면**이다.
 *
 * 심사용으로 줄였다: 기본 정보와 정책 설명만 남긴다. 검증 로그·정책 원문·분위별 지급액은
 * 정책 담당자에게 뜻이 없어 걷었다 (검증은 `/new` 에서 그대로 한다).
 *
 * **run ↔ 정책 매핑은 산출물에 없다 (CONTRACT §7.9).** run_id 만 보고 "무정책 대조군"이라고
 * 쓰면 거짓말이 된다 — `out_BASE` 는 이름과 달리 P010 이 적용된 실행이다.
 * 그래서 이 화면은 **결제 기록(`events.summary` 의 `policy_paid_by_policy_id`)에 남은 지급액**만을
 * 근거로 삼는다. 결제 기록이 없는 실행은 "무정책"이 아니라 **미확인**이다.
 */
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useRun } from '../app/RunContext';
import { READ_ONLY } from '../lib/runtime';
import { Button } from '../components/Button';
import { Card } from '../components/Card';
import { Callout, EmptyState } from '../components/Feedback';
import { loadPolicies, loadPolicyDetail, useLoad } from '../lib/runData';
import { dateRange } from '../lib/format';
import { policyType } from '../lib/labels';

export function PolicyScreen() {
  const navigate = useNavigate();
  // run 은 셸이 소유한다. 이 화면에는 run 선택기를 두지 않는다 (설계도 §5)
  const run = useRun();
  const [descOpen, setDescOpen] = useState(false);

  /** 적용 정책은 결제 기록에서만 읽는다 — 판정 자체는 `RunContext` 가 한다 */
  const binding = run.policy;
  const events = run.bundle.events;
  const applied = binding.items[0] ?? null;

  const policyId = applied?.id ?? null;
  const policies = useLoad('policies', loadPolicies);
  const detailState = useLoad(policyId ? `policy:${policyId}` : null, () => loadPolicyDetail(policyId as string));
  const item = policies.status === 'ready' ? policies.data.items.find((p) => p.id === policyId) : undefined;
  const detail = detailState.status === 'ready' ? detailState.data : undefined;
  // 정책 내용을 받는 중이면 "찾지 못했습니다"를 띄우지 않는다 — 아직 모르는 것과 없는 것은 다르다
  const pending =
    policyId !== null && [policies, detailState].some((s) => s.status === 'loading' || s.status === 'idle');

  if (pending) return <div className="screen-fallback" aria-busy="true" />;

  /* --- 정책이 적용되지 않았거나, 판단할 수 없을 때 -------------------------- */

  if (!applied || !detail || !item) {
    return (
      <div className="stack">
        <header className="pagehead">
          <div className="pagehead__text">
            <h1 className="pagehead__title">
              {binding.known && !applied ? '적용된 정책 없음' : '적용 정책 미확인'}
            </h1>
          </div>
        </header>

        {binding.known && !applied ? (
          /* 결제 기록을 끝까지 읽었는데 정책 지급이 0건 — 이때만 "대조군"이라고 말할 수 있다 */
          <EmptyState
            title="이 실행에는 정책이 적용되지 않았습니다 (대조군)"
            body="결제 기록에 지원금 사용이 없습니다. 정책 없이 돌린 비교 기준입니다."
            action={
              !READ_ONLY && (
              <Button variant="primary" onClick={() => navigate('/new')}>
                정책을 적용해 새로 만들기
              </Button>
              )
            }
          />
        ) : applied ? (
          /* 지급 기록은 있는데 그 식별자의 정책 파일을 찾지 못했다 — 이름을 지어내지 않는다 */
          <EmptyState
            title={`정책 ${applied.id} 의 내용을 찾지 못했습니다`}
            body={`정책 목록에 ${applied.id} 파일이 없습니다.`}
            action={
              !READ_ONLY && (
              <Button variant="primary" onClick={() => navigate('/new')}>
                등록된 정책 보기
              </Button>
              )
            }
          />
        ) : (
          <>
            <Callout tone="warn">
              결제 기록이 없어 적용 정책을 확인할 수 없습니다. 정책이 없었다는 뜻은 아닙니다.
            </Callout>
            <EmptyState
              title="결제 기록이 없습니다"
              body={`사유: ${events.reason ?? '알 수 없습니다.'}`}
              action={
                !READ_ONLY && (
                  <Button variant="primary" onClick={() => navigate('/new')}>
                    새 시뮬레이션 만들기
                  </Button>
                )
              }
            />
          </>
        )}
      </div>
    );
  }

  /* --- 정책이 적용된 실행 --------------------------------------------------- */

  const policy = detail.policy;

  return (
    <div className="stack">
      <header className="pagehead">
        <div className="pagehead__text">
          <h1 className="pagehead__title">{policy.name}</h1>
        </div>
        {!READ_ONLY ? (
          <div className="pagehead__actions">
            <Button onClick={() => navigate('/new')}>이 정책으로 새로 만들기</Button>
          </div>
        ) : null}
      </header>

      <Card title="기본 정보">
        <dl className="dl">
          <div className="dl__item">
            <dt className="dl__k">정책 유형</dt>
            <dd className="dl__v">{policyType(policy.type).label}</dd>
          </div>
          <div className="dl__item">
            <dt className="dl__k">시행 기간</dt>
            <dd className="dl__v num nowrap">{dateRange(policy.effective_from, policy.effective_until)}</dd>
          </div>
          <div className="dl__item">
            <dt className="dl__k">대상 지역</dt>
            <dd className="dl__v">{(policy.target_districts ?? []).join(', ') || '지정 없음'}</dd>
          </div>
          <div className="dl__item">
            <dt className="dl__k">사용처 제한</dt>
            <dd className="dl__v">{item.poi_restricted ? '쿠폰 가맹점에서만 사용' : '제한 없음'}</dd>
          </div>
          <div className="dl__item">
            <dt className="dl__k">사용 업종</dt>
            <dd className="dl__v">{(policy.benefit_categories ?? []).join(', ') || '업종 제한 없음'}</dd>
          </div>
        </dl>

        <div className="stack-sm">
          <p className="dl__k">정책 설명</p>
          <p className={descOpen ? 'card__note wrap' : 'card__note wrap clamp-3'}>{policy.description}</p>
          <div>
            <Button variant="ghost" onClick={() => setDescOpen((v) => !v)}>
              {descOpen ? '설명 접기' : '설명 더 보기'}
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
}
