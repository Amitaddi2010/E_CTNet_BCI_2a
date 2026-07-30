import requests, warnings
warnings.filterwarnings('ignore')

test_urls = [
    'https://bnci-horizon-2020.eu/database/001-2014/true_labels/A01E.mat',
    'https://bnci-horizon-2020.eu/database/data-sets/001-2014/true_labels/A01E.mat',
    'https://lampx.tugraz.at/~bci/database/001-2014/labels/A01E.mat',
    'https://lampx.tugraz.at/~bci/database/001-2014/',
]

for url in test_urls:
    try:
        r = requests.get(url, verify=False, timeout=10, stream=True)
        snippet = r.content[:50]
        clen = r.headers.get('content-length', '?')
        print(f'STATUS {r.status_code} | LEN={clen} | {url}')
        print(f'  first bytes: {snippet}')
    except Exception as e:
        print(f'FAIL: {e} | {url}')
