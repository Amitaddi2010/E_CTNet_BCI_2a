"""
Download BCI Competition IV Dataset 2a MAT files from BNCI Horizon (lampx.tugraz.at).
Uses parallel workers to maximize throughput.
"""
import os, sys, requests, concurrent.futures
from urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

dest_dir = r'F:\Amit\BCI_Classification\C-\Users\Amit_Addi\mne_data\MNE-bnci-data\~bci\database\001-2014'
os.makedirs(dest_dir, exist_ok=True)
base_url = 'https://lampx.tugraz.at/~bci/database/001-2014/'
MIN_SIZE = 30 * 1024 * 1024  # 30 MB threshold for valid files

files = [f'A0{i}{m}.mat' for i in range(1, 10) for m in ['T', 'E']]

def download_file(fname):
    target = os.path.join(dest_dir, fname)
    if os.path.exists(target) and os.path.getsize(target) >= MIN_SIZE:
        print(f'[SKIP] {fname} already downloaded ({os.path.getsize(target)/1024/1024:.1f} MB)', flush=True)
        return True
    # Remove partial file
    if os.path.exists(target):
        os.remove(target)
    url = base_url + fname
    print(f'[START] {fname}...', flush=True)
    try:
        res = requests.get(url, verify=False, stream=True, timeout=60)
        res.raise_for_status()
        with open(target, 'wb') as f:
            total = 0
            for chunk in res.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
        final_size = os.path.getsize(target)
        print(f'[DONE] {fname} ({final_size/1024/1024:.1f} MB)', flush=True)
        return True
    except Exception as e:
        print(f'[ERROR] {fname}: {e}', flush=True)
        if os.path.exists(target):
            os.remove(target)
        return False

print(f'Downloading {len(files)} MAT files with 4 parallel workers...', flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(download_file, files))

done = sum(results)
print(f'\nDownload complete: {done}/{len(files)} files ready.', flush=True)
