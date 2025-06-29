import os
import os.path as osp
from typing import AnyStr

os.environ.setdefault("CHROMATIC_DATADIR", osp.dirname(__file__))


def _load_registry() -> dict[str, str]:
    reg_path = osp.join(os.environ["CHROMATIC_DATADIR"], 'registry.json')
    if not osp.exists(reg_path):
        raise RuntimeError("missing 'registry.json': please reinstall chromatic-python")
    with open(reg_path, mode='rb') as reg:
        import json

        return json.load(reg)


registry = _load_registry()


def filehash(fp: int | AnyStr | os.PathLike[AnyStr], alg='sha256'):
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
    import urllib.request
    from chromatic import __version__

    version = re.sub(r"\.dev0\+.+$", '', __version__)
    remote_dir = f"crypt0lith/chromatic/raw/v{version}/chromatic/data"
    url = f"https://github.com/{remote_dir}/{relpath}"
    return urllib.request.urlretrieve(url, out_path)[0]


def _fetch(basename: str):
    abspath = osp.join(os.environ["CHROMATIC_DATADIR"], basename)
    if osp.exists(abspath) and filehash(abspath) == registry[basename]:
        return abspath
    return _fetch_remote(basename, abspath)


def _load(basename: str):
    import PIL.Image

    return PIL.Image.open(_fetch(basename))
