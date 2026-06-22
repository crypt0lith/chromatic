from os import PathLike
from pathlib import Path
from types import MappingProxyType as mappingproxy


def _load_registry() -> mappingproxy[str, str]:
    reg_fname = "registry.json"
    reg_path = Path(__file__).parent / "images" / reg_fname
    if not reg_path.exists():
        raise RuntimeError(f"missing {reg_fname!r}: please reinstall chromatic-python")
    with reg_path.open("rb") as reg:
        import json

        d = json.load(reg)
    return mappingproxy(d)


registry = _load_registry()


def filehash(fp: str | PathLike[str], alg='sha256'):
    import hashlib

    if alg not in hashlib.algorithms_available:
        raise ValueError(f"unavailable hashing algorithm: {alg!r}")
    hasher = hashlib.new(alg)
    with open(fp, mode='rb') as f:
        chunksize = 0xFFFF + 1
        while chunk := f.read(chunksize):
            hasher.update(chunk)
    return hasher.hexdigest()


def _fetch_remote(relpath: str, out_path: str):
    """Fetch a remote file from the chromatic repo data directory"""
    import re
    import sys
    import urllib.request

    from chromatic import __version__

    version = re.sub(r"\.dev\d+\+.+$", '', __version__)
    remote_dir = f"crypt0lith/chromatic/raw/v{version}/chromatic/data"
    url = f"https://github.com/{remote_dir}/{relpath}"
    print(f"fetching {url!r}...", file=sys.stderr)
    return urllib.request.urlretrieve(url, out_path)[0]


def _fetch(basename: str):
    """Return the absolute path of an image in 'data/images'

    The file is fetched remotely from the chromatic repo
    if it fails the hash check.

    """
    pardir = Path(__file__).parent
    fp = pardir / "images" / basename
    if fp.exists() and filehash(fp) == registry[basename]:
        return str(fp)
    else:
        relpath = fp.relative_to(pardir)
        return _fetch_remote(str(relpath), str(fp))


def _load(basename: str):
    """Load an image from the 'data/images' directory"""
    import PIL.Image

    return PIL.Image.open(_fetch(basename))
