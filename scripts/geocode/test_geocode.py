"""V-WORLD 키 검증 + Geocoder 클라이언트 sanity check."""
import json
import sys
from pathlib import Path

# scripts 디렉토리를 import path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geocode.vworld_client import VWorldGeocoder, load_api_key


def main():
    key = load_api_key()
    print(f"key loaded: {key[:8]}... (len={len(key)})")

    cache_path = Path(__file__).resolve().parent / "cache.sqlite"
    geo = VWorldGeocoder(key, cache_path)

    test_addresses = [
        "서울특별시 종로구 사직로8길 34",          # 경희궁의아침3단지 (K-apt sample)
        "서울특별시 강남구 테헤란로 152",          # 강남파이낸스센터
        "서울특별시 강남구 역삼동 109-16번지",      # 건축물대장 sample (지번)
    ]

    print()
    print("=== Test geocoding 3 addresses ===")
    for addr in test_addresses:
        result = geo.geocode_one(addr, type_first="road", fallback=True)
        print(f"\nInput: {addr}")
        print(f"  status: {result.status}")
        if result.status == "OK":
            print(f"  type_used: {result.type_used}")
            print(f"  lon, lat: {result.lon:.6f}, {result.lat:.6f}")
            print(f"  level1 (시도):     {result.level1}")
            print(f"  level2 (시군구):   {result.level2}")
            print(f"  level3 (법정동):   {result.level3}")
            print(f"  level4A (행정동):  {result.level4A}")
            print(f"  level4AC (행정동코드): {result.level4AC}")
            print(f"  refined: {result.refined}")
        else:
            print(f"  error: {result.raw_error}")

    print()
    print(f"Stats: {geo.stats}")


if __name__ == "__main__":
    main()
