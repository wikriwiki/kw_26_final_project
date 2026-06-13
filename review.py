from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd

# ===== 설정 =====
PLACE_ID = "1285581460"  # 스타벅스 송파개롱역점
URL = f"https://pcmap.place.naver.com/restaurant/{PLACE_ID}/review/visitor"

print("=" * 50)
print("🔍 네이버 지도 리뷰 수집기")
print(f"📍 장소 ID: {PLACE_ID}")
print("=" * 50 + "\n")

# 1. 브라우저 설정
chrome_options = Options()
chrome_options.add_argument("--headless=new")  # 화면 없이 실행 (보고 싶으면 이 줄 주석 처리)
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option("useAutomationExtension", False)

print("🌐 브라우저 시작 중...")
driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=chrome_options,
)
driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
    "source": 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'
})

# 2. 페이지 접속
print(f"📄 페이지 접속 중...")
driver.get(URL)
time.sleep(5)
print("✅ 페이지 로딩 완료!\n")

# 3. "펼쳐서 더보기" 버튼 클릭하여 모든 리뷰 로드
print("📜 리뷰 로딩 중 (더보기 클릭)...")
click_count = 0
max_clicks = 200  # 최대 클릭 횟수 (원하는 만큼 조정. 1클릭당 약 10개 리뷰 로드)

for i in range(max_clicks):
    try:
        # "펼쳐서 더보기" 버튼 (리뷰 목록 하단)
        more_btn = driver.find_element(By.CSS_SELECTOR, "a.fvwqf")
        driver.execute_script("arguments[0].scrollIntoView(true);", more_btn)
        time.sleep(0.3)
        driver.execute_script("arguments[0].click();", more_btn)
        click_count += 1
        time.sleep(1.0)

        if (click_count) % 20 == 0:
            print(f"   ... {click_count}회 클릭 완료")
    except Exception:
        # 버튼이 더 이상 없으면 종료
        break

print(f"   ✅ 더보기 {click_count}회 클릭 완료\n")

# 4. 개별 리뷰의 "더보기" 버튼 클릭 (긴 리뷰 펼치기)
print("📖 긴 리뷰 펼치기...")
try:
    expand_buttons = driver.find_elements(By.CSS_SELECTOR, "a.pui__wFzIYl")
    for btn in expand_buttons:
        try:
            driver.execute_script("arguments[0].click();", btn)
        except:
            pass
    print(f"   ✅ {len(expand_buttons)}개 리뷰 펼침 완료\n")
except:
    print("   리뷰 펼치기 스킵\n")

# 5. 리뷰 데이터 수집
print("📦 리뷰 데이터 수집 중...")

# li.place_apply_pui = 각 리뷰 카드
review_items = driver.find_elements(By.CSS_SELECTOR, "li.place_apply_pui")
print(f"   리뷰 항목 {len(review_items)}개 발견")

reviews_data = []
for idx, item in enumerate(review_items):
    try:
        # 작성자
        try:
            author = item.find_element(By.CSS_SELECTOR, "span.pui__NMi-Dp").text.strip()
        except:
            author = ""

        # 리뷰 본문 (div.pui__vn15t2 안의 a 태그)
        try:
            body = item.find_element(By.CSS_SELECTOR, "div.pui__vn15t2 a").text.strip()
        except:
            body = ""

        # 방문 날짜 (time 태그의 직접 텍스트)
        try:
            visit_date = item.find_element(By.TAG_NAME, "time").text.strip()
        except:
            visit_date = ""

        # 전체 날짜 (pui__blind 중 "2026년..." 형식)
        try:
            blind_spans = item.find_elements(By.CSS_SELECTOR, "span.pui__blind")
            full_date = ""
            for bs in blind_spans:
                t = bs.text.strip()
                if "년" in t and "월" in t and "일" in t:
                    full_date = t
                    break
            visit_date = full_date if full_date else visit_date
        except:
            pass

        # 방문 횟수 & 인증 수단 (span.pui__gfuUIT - "12번째 방문", "영수증")
        visit_count = ""
        verification = ""
        try:
            info_spans = item.find_elements(By.CSS_SELECTOR, "span.pui__gfuUIT")
            for s in info_spans:
                t = s.text.strip()
                if "번째 방문" in t:
                    visit_count = t
                elif t in ("영수증", "네이버 예약", "네이버 주문"):
                    verification = t
        except:
            pass

        # 빈 리뷰 건너뛰기
        if not body and not author:
            continue

        reviews_data.append({
            "작성자": author,
            "리뷰내용": body,
            "방문일": visit_date,
            "방문횟수": visit_count,
            "인증": verification,
        })
    except Exception as e:
        continue

    if (idx + 1) % 100 == 0:
        print(f"   ... {idx + 1}개 처리 완료")

# 6. DataFrame 생성 및 저장
df = pd.DataFrame(reviews_data)
if not df.empty:
    df = df.drop_duplicates(subset=["리뷰내용", "작성자"])

print(f"\n✅ 총 {len(df)}개 리뷰 수집 완료!")

if not df.empty:
    output_file = "naver_reviews.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"💾 '{output_file}'에 저장되었습니다.")
    print(f"\n📋 미리보기 (상위 5개):")
    print(df.head().to_string(index=False))
    print(f"\n📊 통계:")
    print(f"   - 전체 리뷰 수: {len(df)}")
    print(f"   - 리뷰 내용 있는 수: {df['리뷰내용'].str.len().gt(0).sum()}")
    print(f"   - 평균 리뷰 길이: {df['리뷰내용'].str.len().mean():.0f}자")
else:
    print("❌ 수집된 리뷰가 없습니다.")

driver.quit()
print("\n🏁 완료!")