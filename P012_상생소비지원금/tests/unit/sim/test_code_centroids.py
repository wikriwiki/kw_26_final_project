from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from code_centroids import build


def test_same_name_in_distinct_districts_keeps_distinct_coordinates():
    rows = [{'code': '11680510', 'name': '신사동', 'lon': 127.024, 'lat': 37.522},
            {'code': '11620685', 'name': '신사동', 'lon': 126.918, 'lat': 37.485}]
    result = build(rows, 'source')
    assert result['centroids']['11680510'] == [127.024, 37.522]
    assert result['centroids']['11620685'] == [126.918, 37.485]
    assert result['_meta']['name_based_join'] is False


def test_duplicate_code_or_nonfinite_coordinates_are_not_silently_averaged():
    row = {'code': '11680510', 'lon': 127.0, 'lat': 37.5}
    with pytest.raises(ValueError, match='duplicate'): build([row, row], 'source')
    with pytest.raises(ValueError, match='coordinate'): build([dict(row, lon=float('nan'))], 'source')


def test_missing_coordinates_reported_without_borrowing_another_same_name():
    result = build([{'code': '11680510', 'lon': 127., 'lat': 37.5}, {'code': '11620685', 'lon': None, 'lat': None}], 'source')
    assert result['_meta']['missing_coordinate_codes'] == ['11620685']
    assert '11620685' not in result['centroids']
