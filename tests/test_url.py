import pytest
import requests
from cesard.metadata.mapping import URL, DEM_MAP


def url_recursive(key, mapping, parent_key=None):
    key_info = f"{parent_key}.{key}" if parent_key else key
    if isinstance(mapping[key], dict):
        for k, v in mapping[key].items():
            url_recursive(k, mapping[key], key_info)
    else:
        value = mapping[key]
        if value is not None and value.startswith('https'):
            print(key_info)
            response = requests.get(
                url=value,
                headers={"User-Agent": "Mozilla/5.0"},
                allow_redirects=True
            )
            assert response.status_code == 200


@pytest.mark.parametrize('key', URL.keys())
def test_url(key):
    url_recursive(key=key, mapping=URL)


@pytest.mark.parametrize('key', DEM_MAP.keys())
def test_url_dem(key):
    url_recursive(key=key, mapping=DEM_MAP)
