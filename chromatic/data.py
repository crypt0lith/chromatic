import json
import os
from enum import IntEnum
from hashlib import sha256
from json import JSONDecodeError
from os import PathLike
from pathlib import Path

root = Path(os.path.dirname(__file__))
config_fp = root / 'config.json'
stub_fp = root / 'data.pyi'


def _get_checksum(paths: dict[str, str]) -> str:
    return sha256(';'.join(sorted(paths.values())).encode()).hexdigest()


def _build_config_file():
    from PIL.ImageFont import FreeTypeFont

    d = {}
    images_dir, fonts_dir = tuple((n, root / n) for n in ('images', 'fonts'))
    for k, fp in {images_dir, fonts_dir}:
        d[k] = {}
        if k == 'images':
            for fn in os.listdir(fp):
                d[k][os.path.splitext(fn)[0].replace(' ', '_').lower()] = str(
                    (fp / fn).relative_to(root))
        else:
            ttf_dir = fp / 'truetype'
            for fn in os.listdir(ttf_dir):
                abspath = ttf_dir / fn
                try:
                    _ = FreeTypeFont(abspath)
                except OSError as err:
                    err.add_note(str(abspath))
                    raise err
                d[k][os.path.splitext(fn)[0].upper()] = str(abspath.relative_to(root))
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
    sl = [f"__all__ = [{', '.join(map(repr, {'register_user_font', 'UserFont', *img_dict}))}]",
          'from PIL.ImageFile import ImageFile',
          'from enum import IntEnum',
          'from os import PathLike',
          'from pathlib import Path',
          'def register_user_font[AnyStr: (str, bytes)]'
          '(__path: AnyStr | PathLike[AnyStr]) -> None: ...']
    for k in img_dict.keys():
        sl.append(
            f"def {k}() -> ImageFile: ...")
    sl.extend(
        '\n\t'.join(
            ('class UserFont(IntEnum):',
             *map(lambda x: x + ': int', sorted(fonts_dict)),
             '@property',
             'def path(self) -> Path: ...'))
        .replace('\t', ' ' * 4)
        .splitlines())
    with stub_fp.open('w', encoding='utf-8') as fw:
        fw.write('\n'.join(sl))


def _validate_config_file():
    if not config_fp.exists():
        _build_config_file()
        _build_fake_plastic_trees()
    try:
        with config_fp.open('r', encoding='utf-8') as f:
            conf: dict[str, dict[str, str] | str] = json.load(f)
    except JSONDecodeError:
        _build_config_file()
        _build_fake_plastic_trees()
        return _validate_config_file()
    proc_img_path = lambda x: str(x).replace(' ', '_').lower()
    received_checksum = _get_checksum(
        {(proc_img_path if dp.name.endswith('images') else str.upper)(
            os.path.splitext((fn := (dp / fp)).name)[0]): str(
            fn.relative_to(root)) for (dp, df) in
            ((p, os.listdir(p)) for p in (root / 'fonts' / 'truetype', root / 'images')) for fp in
            df})
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
    raise e from None
_image_dict_ = config['images']
_fonts_dict_ = config['fonts']
del _get_checksum, _validate_config_file, _build_config_file, config
if not stub_fp.exists():
    _build_fake_plastic_trees()
del _build_fake_plastic_trees, config_fp, stub_fp


def _create_font_enum() -> type['UserFont']:
    def path(self):
        return Path(_fonts_dict_[self.name])

    enum_cls = IntEnum(
        'UserFont', dict(map(lambda x: (x[-1], x[0] + 1), enumerate(sorted(_fonts_dict_)))))
    enum_cls.path = property(path)
    return enum_cls


UserFont = _create_font_enum()
del _create_font_enum, sha256, IntEnum, json


def register_user_font[AnyStr: (str, bytes)](__path: AnyStr | PathLike[AnyStr]):
    inp_path = getattr(__path, '__fspath__', lambda: __path)()
    if not os.path.exists(inp_path):
        raise FileNotFoundError(
            repr(f"{__path}"))
    fp = Path(__path)
    if fp.is_symlink():
        fp = fp.readlink()
    if fp.suffix != (ttf_ext := '.ttf'):
        raise ValueError(
            f"Expected {ttf_ext!r} file, "
            f"got filetype {fp.suffix!r} instead")
    from PIL.ImageFont import FreeTypeFont
    try:
        _ = FreeTypeFont(fp)
    except OSError as err:
        if fp.exists():
            err.add_note(repr(f"{fp.resolve()}"))
        raise err
    src = fp.absolute()
    dst = Path(os.path.dirname(__file__)) / 'fonts' / 'truetype'
    if not dst.samefile(src if src.is_dir() else src.absolute().parent):
        fname = dst / src.stem
        if dst.drive == src.drive:
            fp = Path(f'{fname}.lnk')
            if fp.exists():
                fp.unlink()
            fp.symlink_to(src)
        else:
            fp = Path(f'{fname}.ttf')
            fp.write_bytes(src.read_bytes())
    ttf_obj = FreeTypeFont(fp)
    print(f"Successfully registered new UserFont: "
          f"{tuple(filter(None, ttf_obj.getname()))!r}")


def __getattr__(name):
    globals_dict = dict(globals())
    if name in globals_dict:
        g_var = globals_dict[name]
        if not (name.startswith('_') and callable(g_var)):
            return g_var
    elif name in _image_dict_:
        from PIL.Image import open as open_img

        return lambda: open_img(_image_dict_[name])
    raise AttributeError(
        f"Module '{__name__}' has no attribute '{name}'")


__all__ = ['UserFont', *_image_dict_]
