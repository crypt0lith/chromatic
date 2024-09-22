import json
import os
from enum import IntEnum
from hashlib import sha256
from pathlib import Path

root = Path(os.path.dirname(__file__))
config_fp = root / 'config.json'
stub_fp = root / 'data.pyi'


def _get_checksum(paths: dict[str, str]) -> str:
    return sha256(';'.join(sorted(paths.values())).encode()).hexdigest()


def _build_config_file():
    d = {}
    images_dir, fonts_dir = tuple((n, root / n) for n in ('images', 'fonts'))
    for k, fp in {images_dir, fonts_dir}:
        d[k] = {}
        if k == 'images':
            for fn in os.listdir(fp):
                d[k][os.path.splitext(fn)[0].replace(' ', '_').lower()] = str((fp / fn).relative_to(root))
        else:
            ttf_dir = fp / 'truetype'
            for fn in os.listdir(ttf_dir):
                d[k][os.path.splitext(fn)[0].upper()] = str((ttf_dir / fn).relative_to(root))
    d['checksum'] = _get_checksum(d['images'] | d['fonts'])
    with config_fp.open('w', encoding='utf-8') as f:
        json.dump(d, f, indent='\t')


_image_dict_: dict[str, str] = {}
_fonts_dict_: dict[str, str] = {}


def _build_fake_plastic_trees():
    img_dict = _image_dict_
    fonts_dict = _fonts_dict_
    if not img_dict:
        with config_fp.open('r') as f:
            img_dict = json.load(f)['images']
    if not fonts_dict:
        with config_fp.open('r') as f:
            fonts_dict = json.load(f)['fonts']
    sl = [f"__all__ = [{', '.join(repr(k) for k in {'UserFont', *img_dict})}]",
          'from PIL.ImageFile import ImageFile',
          'from enum import IntEnum',
          'from pathlib import PurePath']
    for k in img_dict.keys():
        sl.append(
            f"def {k}() -> ImageFile: ...")
    sl.extend(
        '\n\t'.join(
            ('class UserFont(IntEnum):',
             *map(lambda x: x + ': int', sorted(fonts_dict)),
             '@property',
             'def path(self) -> PurePath: ...'))
        .replace('\t', ' ' * 4)
        .splitlines())
    with stub_fp.open('w', encoding='utf-8') as fw:
        fw.write('\n'.join(sl))


def _validate_config_file():
    if not config_fp.exists():
        _build_config_file()
        _build_fake_plastic_trees()
    with config_fp.open('r', encoding='utf-8') as f:
        conf: dict[str, dict[str, str] | str] = json.load(f)
    proc_img_path = lambda x: str(x).replace(' ', '_').lower()
    received_checksum = _get_checksum(
        {(proc_img_path if dp.name.endswith('images') else str.upper)(
            os.path.splitext((fn := (dp / fp)).name)[0]): str(fn.relative_to(root)) for (dp, df) in
         ((p, os.listdir(p)) for p in (root / 'fonts' / 'truetype', root / 'images')) for fp in df})
    if received_checksum != conf.get('checksum'):
        if __name__ == '__main__':
            print('Received hash does not match checksum\nRebuilding data manifest...')
        config_fp.unlink()
        if stub_fp.exists():
            stub_fp.unlink()
        return _validate_config_file()
    for subdir in {'images', 'fonts'}:
        conf[subdir] = {k: str(root / v) for k, v in conf.get(subdir, {}).items()}
    return conf


try:
    config = _validate_config_file()
except ImportError as e:
    raise ImportError(
        e) from None
_image_dict_ = config['images']
_fonts_dict_ = config['fonts']
del _get_checksum, _validate_config_file, _build_config_file, root, config
if not stub_fp.exists():
    _build_fake_plastic_trees()
del _build_fake_plastic_trees, config_fp, stub_fp


def _create_font_enum() -> type['UserFont']:
    def path(self):
        from pathlib import PurePath

        return PurePath(_fonts_dict_[self.name])

    enum_cls = IntEnum('UserFont', dict(map(lambda x: (x[-1], x[0] + 1), enumerate(sorted(_fonts_dict_)))))
    enum_cls.path = property(path)
    return enum_cls


UserFont = _create_font_enum()
del _create_font_enum, sha256, Path, IntEnum, json, os


def __getattr__(name):
    globals_dict = dict(globals())
    if name in globals_dict:
        return globals_dict[name]
    if name in _image_dict_:
        from PIL.Image import open as open_img

        return lambda: open_img(_image_dict_[name])
    raise AttributeError(
        f"Module '{__name__}' has no attribute '{name}'")


__all__ = ['UserFont', *_image_dict_]
