/**
 * /__kit — **개발 전용** 컴포넌트 전시.
 *
 * 사용자 내비게이션에 넣지 않는다 (스펙 §7). 앱 셸 밖에 있는 별도 라우트라
 * 주소를 직접 치지 않으면 도달할 수 없다.
 */
import { useState } from 'react';
import { Badge } from '../components/Badge';
import { Button } from '../components/Button';
import { Card, Stat } from '../components/Card';
import { Disclosure } from '../components/Disclosure';
import { Callout, EmptyState, ErrorState, SkeletonCard, SkeletonText } from '../components/Feedback';
import { SelectField, TextAreaField, TextField } from '../components/Field';
import { AlertTriangleIcon, RefreshIcon } from '../components/Icon';
import { BarList, Meter } from '../components/Meter';

const COLORS = [
  ['--bg', '페이지 배경'],
  ['--surface', '카드'],
  ['--surface-sunken', '표 헤더·입력'],
  ['--fg', '본문 17.1:1'],
  ['--fg-muted', '보조 7.2:1'],
  ['--border', '헤어라인'],
  ['--border-strong', '입력 테두리'],
  ['--primary', '주요 동작'],
  ['--ring', '포커스 링'],
  ['--ok', '정상'],
  ['--warn', '확인 필요'],
  ['--danger', '오류'],
  ['--info', '안내'],
];

const TYPE_SCALE: Array<[string, string, string]> = [
  ['32px', 'var(--fs-2xl)', '화면 제목'],
  ['24px', 'var(--fs-xl)', '화면 제목 (좁은 폭)'],
  ['20px', 'var(--fs-lg)', '섹션 제목'],
  ['16px', 'var(--fs-body)', '본문 — 기본'],
  ['14px', 'var(--fs-md)', '표·버튼'],
  ['13px', 'var(--fs-sm)', '보조 설명'],
  ['12px', 'var(--fs-caption)', '캡션·레이블 전용'],
];

export function KitScreen() {
  const [busy, setBusy] = useState(false);

  return (
    <div className="page">
      <div className="stack">
        <header className="pagehead">
          <div className="pagehead__text">
            <p className="pagehead__eyebrow">개발 전용</p>
            <h1 className="pagehead__title">컴포넌트 전시</h1>
            <p className="pagehead__purpose">
              토큰과 컴포넌트가 스펙대로 그려지는지 확인하는 개발용 화면입니다.
            </p>
          </div>
        </header>

        <p className="kit__banner">
          <AlertTriangleIcon size={18} />
          <span>
            이 화면은 사용자 내비게이션에 없습니다. 제품 화면(정책 설정·실행 모니터·결과)에는 이
            페이지의 내용이 들어가지 않습니다.
          </span>
        </p>

        <section className="section">
          <h2 className="section__title">색</h2>
          <div className="kit__swatches">
            {COLORS.map(([token, use]) => (
              <figure className="kit__swatch" key={token}>
                <div className="kit__chip" style={{ background: `var(${token})` }} />
                <figcaption>
                  {token}
                  <br />
                  {use}
                </figcaption>
              </figure>
            ))}
          </div>
        </section>

        <section className="section">
          <h2 className="section__title">타이포</h2>
          <div className="kit__type">
            {TYPE_SCALE.map(([px, varName, use]) => (
              <div className="kit__type-row" key={px}>
                <span className="kit__type-tag">{px}</span>
                <span style={{ fontSize: varName }}>정책 시뮬레이션 콘솔 1,234,567원</span>
                <span className="card__note">{use}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="section">
          <h2 className="section__title">버튼</h2>
          <div className="row">
            <Button variant="primary">기본 동작</Button>
            <Button variant="secondary">보조 동작</Button>
            <Button variant="ghost">약한 동작</Button>
            <Button variant="primary" disabled>
              비활성
            </Button>
            <Button
              variant="primary"
              icon={<RefreshIcon size={18} />}
              busy={busy}
              busyLabel="처리 중"
              onClick={() => {
                setBusy(true);
                window.setTimeout(() => setBusy(false), 1200);
              }}
            >
              비동기 동작
            </Button>
          </div>
        </section>

        <section className="section">
          <h2 className="section__title">입력</h2>
          <div className="grid">
            <div className="c4">
              <TextField label="정책 이름" defaultValue="민생회복 소비쿠폰 1차" help="공고문에 쓰인 이름 그대로 입력합니다." />
            </div>
            <div className="c4">
              <TextField
                label="1분위 지급액"
                defaultValue="0"
                error="지급액은 1원 이상이어야 합니다 — 금액을 다시 입력하세요."
              />
            </div>
            <div className="c4">
              <SelectField
                label="지급 기준"
                defaultValue="spend_decile"
                options={[
                  { value: 'spend_decile', label: '소비 10분위' },
                  { value: 'income', label: '소득 구간' },
                ]}
                help="파일에 값이 없으면 소득 구간이 기본값으로 적용됩니다."
              />
            </div>
            <div className="c12">
              <TextAreaField
                label="정책 설명"
                defaultValue="대상자가 하루를 계획할 때 읽는 문장입니다."
                help="여기에 쓴 문장이 그대로 대상자에게 전달됩니다."
              />
            </div>
          </div>
        </section>

        <section className="section">
          <h2 className="section__title">뱃지 · 지표</h2>
          <div className="row">
            <Badge>중립</Badge>
            <Badge tone="ok">완료</Badge>
            <Badge tone="warn">확인 필요</Badge>
            <Badge tone="danger">오류</Badge>
            <Badge tone="info">안내</Badge>
          </div>
          <div className="grid">
            <div className="card c3">
              <Stat label="처리한 대상자" value="1,400" unit="명" hint="7일 합계" />
            </div>
            <div className="card c3">
              <Stat label="총 소요 시간" value="1시간 30분" />
            </div>
            <div className="card c3">
              <Stat label="목표 인원" value={null} unknownReason="요약 파일이 없습니다" />
            </div>
            <div className="card c3">
              <Stat label="쿠폰 사용처 비율" value="99.2%" hint="7,725건 / 7,785건" />
            </div>
          </div>
        </section>

        <section className="section">
          <h2 className="section__title">표 · 막대</h2>
          <div className="grid">
            <Card className="c6" flush>
              <div className="table-wrap">
                <table className="table table--lead">
                  <thead>
                    <tr>
                      <th scope="col">일자</th>
                      <th scope="col" className="n">
                        처리 인원
                      </th>
                      <th scope="col" className="n col-md">
                        소요
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <th scope="row" className="num" style={{ fontWeight: 400 }}>
                        2025-07-21
                      </th>
                      <td className="n">200</td>
                      <td className="n col-md">11분 4초</td>
                    </tr>
                    <tr>
                      <th scope="row" className="num" style={{ fontWeight: 400 }}>
                        2025-07-22
                      </th>
                      <td className="n">200</td>
                      <td className="n col-md">11분 16초</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </Card>
            <Card className="c6" title="막대">
              <Meter ratio={0.72} label="예시 진행률" className="progress__track" />
              <BarList
                items={[
                  { key: 'a', name: '식사', value: 11892201, display: '1,189만원' },
                  { key: 'b', name: '카페', value: 6120000, display: '612만원' },
                  { key: 'c', name: '의류', value: 1044500, display: '104만원' },
                ]}
              />
            </Card>
          </div>
        </section>

        <section className="section">
          <h2 className="section__title">로딩 · 빈 상태 · 오류</h2>
          <div className="grid">
            <div className="c4">
              <SkeletonCard />
            </div>
            <div className="c4">
              <Card>
                <SkeletonText lines={4} />
              </Card>
            </div>
            <div className="c4">
              <EmptyState
                title="아직 실행한 시뮬레이션이 없습니다"
                body="정책을 검증한 뒤 실행하면 여기에 진행 상황이 나타납니다."
                action={<Button variant="primary">정책 검증하러 가기</Button>}
              />
            </div>
            <div className="c6">
              <ErrorState
                title="결과를 불러오지 못했습니다"
                body="서버가 응답하지 않습니다. 잠시 뒤 다시 시도하거나, 실행 중인 서버 주소를 확인하세요."
                action={
                  <Button variant="secondary" icon={<RefreshIcon size={18} />}>
                    다시 시도
                  </Button>
                }
                detail={'GET /api/runs\nfailed to fetch'}
              />
            </div>
            <div className="c6">
              <div className="stack-sm">
                <Callout>참고용 값입니다. 정본 기록이 있으면 항상 그쪽이 맞습니다.</Callout>
                <Callout tone="warn">
                  소비 분위를 알 수 없는 대상자가 5명 있습니다. 합계가 맞지 않을 수 있습니다.
                </Callout>
              </div>
            </div>
          </div>
        </section>

        <section className="section">
          <h2 className="section__title">접기</h2>
          <div>
            <Disclosure title="검증 상세 로그" meta="17건">
              <p className="card__note">기본은 닫힘. 상세·원문은 전부 이 안으로 들어간다.</p>
            </Disclosure>
            <Disclosure title="원문 보기">
              <pre className="code">{'{\n  "id": "P010",\n  "grant_key": "spend_decile"\n}'}</pre>
            </Disclosure>
          </div>
        </section>
      </div>
    </div>
  );
}
