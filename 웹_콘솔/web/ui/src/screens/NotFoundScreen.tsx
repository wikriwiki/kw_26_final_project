import { useNavigate } from 'react-router-dom';
import { Button } from '../components/Button';

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
      <div>
        <Button variant="primary" onClick={() => navigate('/')}>
          처음 화면으로
        </Button>
      </div>
    </div>
  );
}
