from os import PathLike
from pathlib import Path


def _load_registry() -> dict[str, str]:
    reg_fname = "registry.json"
    reg_path = Path(__file__).parent / reg_fname
    if not reg_path.exists():
        raise RuntimeError(f"missing {reg_fname!r}: please reinstall chromatic-python")
    with reg_path.open("rb") as reg:
        import json

        return json.load(reg)


registry = _load_registry()


def filehash(fp: str | PathLike[str], alg='sha256'):
    import hashlib

    if alg not in hashlib.algorithms_available:
        raise ValueError(f"unavailable hashing algorithm: {alg!r}")
    chunksize = 0xFFFF + 1
    hasher = hashlib.new(alg)
    with open(fp, mode='rb') as f:
        while chunk := f.read(chunksize):
            hasher.update(chunk)
    return hasher.hexdigest()


def _fetch_remote(relpath: str, out_path: str):
    import re
    import sys
    import urllib.request

    from chromatic import __version__

    version = re.sub(r"\.dev0\+.+$", '', __version__)
    remote_dir = f"crypt0lith/chromatic/raw/v{version}/chromatic/data"
    url = f"https://github.com/{remote_dir}/{relpath}"
    print(f"fetching {url!r}...", file=sys.stderr)
    return urllib.request.urlretrieve(url, out_path)[0]


def _fetch(basename: str):
    fp = Path(__file__).parent / basename
    if fp.exists() and filehash(fp) == registry[basename]:
        return str(fp)
    else:
        return _fetch_remote(basename, str(fp))


def _load(basename: str):
    import PIL.Image

    return PIL.Image.open(_fetch(basename))
