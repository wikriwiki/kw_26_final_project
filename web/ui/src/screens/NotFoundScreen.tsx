import { useNavigate } from 'react-router-dom';
import { Button } from '../components/Button';
import { EmptyState } from '../components/Feedback';

export function NotFoundScreen() {
  const navigate = useNavigate();
  return (
    <div className="stack">
      <header className="pagehead">
        <div className="pagehead__text">
          <h1 className="pagehead__title">찾는 화면이 없습니다</h1>
          <p className="pagehead__purpose">주소가 바뀌었거나 잘못 입력됐을 수 있습니다.</p>
        </div>
      </header>
      <EmptyState
        title="이 주소에 해당하는 화면이 없습니다"
        body="왼쪽 메뉴에서 정책 설정 · 실행 모니터 · 결과 중 하나를 고르세요."
        action={
          <Button variant="primary" onClick={() => navigate('/')}>
            정책 설정으로 가기
          </Button>
        }
      />
    </div>
  );
}
