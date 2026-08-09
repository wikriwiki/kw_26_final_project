/**
 * 이 실행에 적용된 정책 — 설계도 §5, 라우트 `/runs/:runId/policy`.
 *
 * **화면의 성격이 바뀌었다.** 전에는 "정책을 골라 검증하는" 편집 화면이었지만,
 * 정책을 고르고 검증하는 일은 `/new`(새 시뮬레이션 만들기)로 옮겼다 — 정책이 실제로 쓰이는 시점이다.
 * 여기는 **이미 끝난(또는 도는) 실행에 무엇이 적용됐는지 확인하는 읽기 화면**이다.
 *
 * 그래서 걷어낸 것: 정책 목록에서 고르는 표, 화면 안의 run 선택기, 검증 실행 버튼.
 * 남긴 것: 기본 정보 · 분위별 지급액 · 사용처 제한 · 정책 설명 · 검증 결과.
 *
 * **run ↔ 정책 매핑은 산출물에 없다 (CONTRACT §7.9).** run_id 만 보고 "무정책 대조군"이라고
 * 쓰면 거짓말이 된다 — `out_BASE` 는 이름과 달리 P010 이 적용된 실행이다.
 * 그래서 이 화면은 **결제 기록(`events.summary` 의 `policy_paid_by_policy_id`)에 남은 지급액**만을
 * 근거로 삼고, 그 근거가 무엇인지 화면에 적는다. 결제 기록이 없는 실행은 "무정책"이 아니라
 * **미확인**이다.
 */
import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useRun } from '../app/RunContext';
import { Button } from '../components/Button';
import { Card } from '../components/Card';
import { Disclosure } from '../components/Disclosure';
import { Callout, EmptyState } from '../components/Feedback';
import {
  AlertCircleIcon,
  AlertTriangleIcon,
  CheckCircleIcon,
} from '../components/Icon';
import { BarList } from '../components/Meter';
import type { BarItem } from '../components/Meter';
import { policiesIndex, policyDetails, policyValidations } from '../lib/fixtures';
import { int, krw } from '../lib/format';
import { grantKey, policyType } from '../lib/labels';

const INCOME_ORDER = ['하', '중하', '중', '중상', '상'];

type Grade = 'pass' | 'warn' | 'fail';

/** 등급별 아이콘. 색은 `check--lead` 가 붙은 항목에만 들어간다 */
function GradeIcon({ grade }: { grade: Grade }) {
  if (grade === 'pass') return <CheckCircleIcon size={16} />;
  if (grade === 'warn') return <AlertTriangleIcon size={16} />;
  return <AlertCircleIcon size={16} />;
}

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
  const item = policyId ? policiesIndex.items.find((p) => p.id === policyId) : undefined;
  const detail = policyId ? policyDetails[policyId] : undefined;
  const validation = policyId ? policyValidations[policyId] : undefined;

  const grantBars = useMemo<BarItem[]>(() => {
    if (!detail) return [];
    const decileGrants = detail.policy.decile_grants;
    if (decileGrants && Object.keys(decileGrants).length > 0) {
      return Object.entries(decileGrants)
        .map(([k, v]) => ({ key: k, name: `${k}분위`, value: v, display: krw(v) }))
        .sort((a, b) => Number(a.key) - Number(b.key));
    }
    const incomeGrants = detail.policy.income_grants;
    if (incomeGrants && Object.keys(incomeGrants).length > 0) {
      return Object.entries(incomeGrants)
        .map(([k, v]) => ({ key: k, name: `소득 ${k}`, value: v, display: krw(v) }))
        .sort((a, b) => INCOME_ORDER.indexOf(a.key) - INCOME_ORDER.indexOf(b.key));
    }
    return [];
  }, [detail]);

  /* --- 정책이 적용되지 않았거나, 판단할 수 없을 때 -------------------------- */

  if (!applied || !detail || !item || !validation) {
    return (
      <div className="stack">
        <header className="pagehead">
          <div className="pagehead__text">
            <h1 className="pagehead__title">
              {binding.known && !applied ? '적용된 정책 없음' : '적용 정책 미확인'}
            </h1>
            <p className="pagehead__purpose">
              이 실행에 적용된 정책의 내용입니다. 여기서는 바꿀 수 없습니다.
            </p>
          </div>
        </header>

        {binding.known && !applied ? (
          /* 결제 기록을 끝까지 읽었는데 정책 지급이 0건 — 이때만 "대조군"이라고 말할 수 있다 */
          <EmptyState
            title="이 실행에는 정책이 적용되지 않았습니다 (대조군)"
            body="결제 기록에 정책 지갑에서 빠져나간 금액이 한 건도 없습니다. 정책 없이 돌린 비교 기준 실행입니다."
            action={
              <Button variant="primary" onClick={() => navigate('/new')}>
                정책을 적용해 새로 만들기
              </Button>
            }
          />
        ) : applied ? (
          /* 지급 기록은 있는데 그 식별자의 정책 파일을 찾지 못했다 — 이름을 지어내지 않는다 */
          <EmptyState
            title={`정책 ${applied.id} 의 내용을 찾지 못했습니다`}
            body={`이 실행의 결제 기록에는 ${applied.id} 지급이 남아 있지만, 등록된 정책 목록에 같은 식별자가 없습니다. 정책 파일이 옮겨졌거나 이름이 바뀌었을 수 있습니다.`}
            action={
              <Button variant="primary" onClick={() => navigate('/new')}>
                등록된 정책 보기
              </Button>
            }
          />
        ) : (
          <>
            <Callout tone="warn">
              이 실행에 정책이 적용됐는지 확인할 수 없습니다. 실행 산출물에는 적용 정책 식별자가
              따로 기록되지 않아 결제 기록에 남은 지급액으로만 판단하는데, 이 실행에는 그 기록이
              없습니다. <strong>정책이 없었다는 뜻이 아니라, 아직 모른다는 뜻입니다.</strong>
            </Callout>
            <EmptyState
              title="결제 기록이 없습니다"
              body={`사유: ${events.reason ?? '알 수 없습니다.'}`}
              action={
                <Button variant="primary" onClick={() => navigate('/new')}>
                  새 시뮬레이션 만들기
                </Button>
              }
            />
          </>
        )}
      </div>
    );
  }

  /* --- 정책이 적용된 실행 --------------------------------------------------- */

  const policy = detail.policy;
  const effectiveKey = grantKey(detail.grant_key_effective);
  const excluded = (policy.excluded_income ?? []).concat(
    (policy.excluded_deciles ?? []).map((d) => `${d}분위`),
  );

  const failing = validation.checks.filter((c) => c.grade === 'fail');
  const warning = validation.checks.filter((c) => c.grade === 'warn');
  const passing = validation.checks.filter((c) => c.grade === 'pass');

  /** 이 화면에서 색을 쓸 등급 하나. 나머지는 무채색 아이콘으로 둔다 (§7b) */
  const lead: Grade = failing.length > 0 ? 'fail' : warning.length > 0 ? 'warn' : 'pass';

  const paidHere = events.policy_paid_by_policy_id?.[applied.id] ?? null;
  const otherIds = binding.items.slice(1).map((p) => p.id);

  return (
    <div className="stack">
      <header className="pagehead">
        <div className="pagehead__text">
          <h1 className="pagehead__title">{policy.name}</h1>
          <p className="pagehead__purpose">
            이 실행에 적용된 정책의 내용입니다. 여기서는 바꿀 수 없습니다.
          </p>
        </div>
        <div className="pagehead__actions">
          <Button onClick={() => navigate('/new')}>이 정책으로 새로 만들기</Button>
        </div>
      </header>

      {/* 1차 블록 1 — 무엇을 근거로 "적용됐다"고 말하는가 */}
      <p className="section__note wrap">
        이 실행의 결제 기록에 <span className="num">{policy.id}</span> 지급{' '}
        <span className="num">{krw(paidHere)}</span>이 남아 있어 적용 정책으로 판단했습니다. 실행
        산출물에는 정책 식별자가 따로 기록되지 않아, 결제 기록이 유일한 근거입니다.
        {otherIds.length > 0
          ? ` 같은 실행에서 다른 정책(${otherIds.join(', ')})의 지급도 함께 확인됩니다.`
          : ''}
      </p>

      {/* 1차 블록 2·3 — 기본 정보 / 지급액 */}
      <div className="grid">
        <Card className="c6" title="기본 정보">
          <dl className="dl">
            <div className="dl__item">
              <dt className="dl__k">정책 유형</dt>
              <dd className="dl__v">{policyType(policy.type).label}</dd>
            </div>
            <div className="dl__item">
              <dt className="dl__k">시행 기간</dt>
              <dd className="dl__v num">
                {policy.effective_from} ~ {policy.effective_until}
              </dd>
            </div>
            <div className="dl__item">
              <dt className="dl__k">대상 지역</dt>
              <dd className="dl__v">{(policy.target_districts ?? []).join(', ') || '지정 없음'}</dd>
            </div>
            <div className="dl__item">
              <dt className="dl__k">사용처 제한</dt>
              <dd className="dl__v">
                {item.poi_restricted ? '쿠폰 가맹점에서만 사용' : '제한 없음'}
              </dd>
            </div>
            <div className="dl__item">
              <dt className="dl__k">지급 기준</dt>
              <dd className="dl__v">
                {effectiveKey.label}
                {detail.grant_key_source === 'default' ? (
                  <span className="cell-sub">정책 파일에 없어 기본값을 적용했습니다</span>
                ) : null}
              </dd>
            </div>
            <div className="dl__item">
              <dt className="dl__k">사용 업종</dt>
              <dd className="dl__v">
                {(policy.benefit_categories ?? []).join(', ') || '업종 제한 없음'}
              </dd>
            </div>
          </dl>

          <div className="stack-sm">
            <p className="dl__k">정책 설명</p>
            <p className={descOpen ? 'card__note wrap' : 'card__note wrap clamp-3'}>
              {policy.description}
            </p>
            <div>
              <Button variant="ghost" onClick={() => setDescOpen((v) => !v)}>
                {descOpen ? '설명 접기' : '설명 더 보기'}
              </Button>
            </div>
          </div>
        </Card>

        <Card
          className="c6"
          title="분위별 지급액"
          note={
            grantBars.length > 0
              ? `${effectiveKey.label} ${grantBars.length}구간${
                  excluded.length > 0 ? ` · 지급 제외 ${excluded.join(', ')}` : ''
                }`
              : undefined
          }
        >
          {grantBars.length > 0 ? (
            <BarList items={grantBars} />
          ) : (
            <EmptyState
              fill
              title="지급액이 없는 정책입니다"
              body={`${policyType(policy.type).label} 정책이라 대상자에게 지급하는 금액이 없습니다. 효과는 소비 금액이 아니라 방문·체류 변화로 나타납니다.`}
            />
          )}
        </Card>
      </div>

      {/* 1차 블록 4 — 검증 결과 (읽기 전용. 이 화면에서 다시 돌리지 않는다) */}
      <section className="section">
        <div className="section__head">
          <h2 className="section__title">검증 결과</h2>
          <p className="section__note">
            정책 파일이 시뮬레이션에 그대로 배선되는지 확인한 사전 점검 결과입니다.
          </p>
        </div>
        <Card>
          <div className="verdict">
            {failing.length > 0 ? (
              <AlertCircleIcon size={24} style={{ color: 'var(--danger)' }} />
            ) : (
              <CheckCircleIcon size={24} style={{ color: 'var(--ok)' }} />
            )}
            <span className="verdict__text">
              {failing.length > 0
                ? `오류 ${int(failing.length)}건이 있는 상태로 실행됐습니다`
                : '점검에서 오류가 나오지 않은 정책입니다'}
            </span>
          </div>

          {/* 뱃지 세 개 대신 한 줄 (§7b) */}
          <p className="tally">
            <span>
              통과 <span className="tally__n">{int(passing.length)}</span>
            </span>
            <span>
              확인 필요 <span className="tally__n">{int(warning.length)}</span>
            </span>
            <span>
              오류 <span className="tally__n">{int(failing.length)}</span>
            </span>
          </p>

          {failing.length + warning.length === 0 ? (
            <p className="card__note">확인이 필요한 항목이 없습니다.</p>
          ) : (
            <ul className="checks">
              {failing.map((c) => (
                <li
                  className={`check check--fail${lead === 'fail' ? ' check--lead' : ''}`}
                  key={`f-${c.message}`}
                >
                  <GradeIcon grade="fail" />
                  <span className="wrap">{c.message}</span>
                </li>
              ))}
              {warning.map((c) => (
                <li
                  className={`check check--warn${lead === 'warn' ? ' check--lead' : ''}`}
                  key={`w-${c.message}`}
                >
                  <GradeIcon grade="warn" />
                  <span className="wrap">{c.message}</span>
                </li>
              ))}
            </ul>
          )}

          {validation.verdict ? (
            <p className="card__note">검증기 판정 원문: {validation.verdict}</p>
          ) : null}

          {!validation.db_wiring_checked ? (
            <p className="card__note wrap">
              데이터베이스 연결 정보가 없어 “정책이 실제로 대상자에게 보이는지”는 확인하지
              못했습니다. 이 항목은 통과가 아니라 <strong>미확인</strong>입니다.
            </p>
          ) : null}
        </Card>
      </section>

      {/* 접어두는 것 — 기본 닫힘 */}
      <section className="section">
        <h2 className="section__title">상세 보기</h2>
        <div>
          <Disclosure title="검증 상세 로그" meta={`${validation.checks.length}건`}>
            <ul className="checks">
              {validation.checks.map((c, i) => (
                <li
                  className={`check check--${c.grade}${c.grade === lead ? ' check--lead' : ''}`}
                  key={`${c.grade}-${i}`}
                >
                  <GradeIcon grade={c.grade as Grade} />
                  <span className="wrap">{c.message}</span>
                </li>
              ))}
            </ul>
            <p className="card__note wrap">
              실행 명령: <span className="num">{validation.command.join(' ')}</span>
            </p>
          </Disclosure>

          <Disclosure
            title="대상자에게 전달되는 문장 (프롬프트 미리보기)"
            meta={validation.prompt_preview_persona ?? undefined}
          >
            <p className="card__note">
              시뮬레이션 대상자가 하루를 계획할 때 실제로 읽는 문장입니다.
            </p>
            <pre className="code">{validation.prompt_preview}</pre>
          </Disclosure>

          <Disclosure title="정책 원문 (JSON)">
            <p className="card__note">
              저장된 파일 {detail.file} 의 내용 그대로입니다. 화면이 요약하며 생략한 값이 있는지
              확인할 때 씁니다.
            </p>
            <pre className="code">{JSON.stringify(policy, null, 2)}</pre>
          </Disclosure>
        </div>
      </section>
    </div>
  );
}
