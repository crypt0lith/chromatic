import json
import os
from enum import IntEnum
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Literal, TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from _typeshed import SupportsWrite

data_root = Path(os.path.dirname(__file__))
stub = data_root / (__file__ + 'i')
config = data_root / 'config.json'
fonts = data_root / 'fonts'
images = data_root / 'images'


def _get_checksum(__fp_iter: Iterable[str]) -> str:
    return sha256(';'.join(sorted(__fp_iter)).encode()).hexdigest()


def _is_img_ext(__fp: str):
    return __fp.endswith(('.png', '.jpg', '.jpeg', '.webp'))


def _is_ttf_ext(__fp: str):
    return __fp.endswith(('.ttf', '.ttc'))


def _build_config():
    from string import printable

    f: SupportsWrite
    d: dict[Literal['fonts', 'images'] | Literal['__hash__'], dict[str, str] | str]
    d = {'fonts': {}, 'images': {}, '__hash__': ''}
    printable = ''.join(c for c in printable if any((c.isalnum(), c.isidentifier(), c == '.')))
    for font_fp in ((Path(fonts) / x).absolute() for x in filter(_is_ttf_ext, os.listdir(fonts))):
        font_name = font_fp.stem
        for c in set(font_name):
            if c not in printable:
                font_name = font_name.replace(c, '_')
        d['fonts'][font_name.upper()] = str(font_fp.relative_to(data_root))
    for image_fp in (
        (Path(images) / x).absolute() for x in filter(_is_img_ext, os.listdir(images))
    ):
        img_name = image_fp.stem
        for c in set(img_name):
            if c not in printable:
                img_name = img_name.replace(c, '_')
        d['images'][img_name.lower()] = str(image_fp.relative_to(data_root))
    d['__hash__'] = _get_checksum((d['fonts'] | d['images']).values())
    with config.open('w', encoding='utf-8') as f:
        json.dump(d, f, indent='\t')
    return d


def _validate(**kwargs):
    need_new_stub = kwargs.get('update_stub', False)
    if need_new_stub or not config.exists():
        json_data = _build_config()
        need_new_stub = True
    else:
        json_data = json.load(config.open('r'))
        s1, s2 = (
            set(
                str(fp.relative_to(data_root))
                for fp in ((Path(subdir) / x).absolute() for x in filter(f, os.listdir(subdir)))
            )
            for f, subdir in zip((_is_ttf_ext, _is_img_ext), (fonts, images))
        )
        if json_data['__hash__'] != _get_checksum(s1 | s2):
            json_data = _build_config()
            need_new_stub = True
    font_data, img_data = json_data['fonts'], json_data['images']
    if need_new_stub or not stub.exists():
        exports = ['register_user_font', 'UserFont']
        img_funcdefs = []
        for img_name in img_data.keys():
            exports.append(img_name)
            img_funcdefs.append(f"def {img_name}() -> ImageFile: ...")
        body = [
            f"__all__ = [{', '.join(map(repr, exports))}]",
            'from PIL.ImageFile import ImageFile',
            'from enum import IntEnum',
            'from os import PathLike',
            'from pathlib import Path',
            'def register_user_font[AnyStr: (str, bytes)]'
            '(__path: AnyStr | PathLike[AnyStr]) -> None: ...',
        ] + img_funcdefs
        body.extend(
            '\n\t'.join(
                (
                    'class UserFont(IntEnum):',
                    *(f'{x} = {i}' for i, x in enumerate(sorted(font_data))),
                    '@property',
                    'def path(self) -> Path: ...',
                )
            )
            .replace('\t', ' ' * 4)
            .splitlines()
        )
        stub.open('w', encoding='utf-8').write('\n'.join(body))
    return font_data, img_data


_font_data_, _img_data_ = _validate()


def _create_font_enum() -> type['UserFont']:
    def path(self):
        return data_root / Path(_font_data_[self.name])

    enum_cls = IntEnum('UserFont', {k: i for (i, k) in enumerate(sorted(_font_data_.keys()))})
    enum_cls.path = property(path)
    return enum_cls


UserFont = _create_font_enum()


def register_user_font[AnyStr: (str, bytes)](__path: AnyStr | os.PathLike[AnyStr]):
    """Register a .ttf font file as a new :class:`UserFont` enum member.

    If the source file is on the same drive :mod:`chromatic.data`,
    a symlink to it is added to the ``data/fonts`` package subdirectory.

    Parameters
    ----------
    __path : AnyStr | PathLike[AnyStr]
        Path to the .ttf font file being registered.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    ValueError
        If the file is not a .ttf font.
    OSError
        If the font file is invalid or cannot be loaded.
    """
    path = os.fspath(__path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{__path!r}")
    path_obj = Path(path)
    if path_obj.is_symlink():
        path_obj = path_obj.readlink()
    if path_obj.suffix != '.ttf':
        raise ValueError(f"Expected '.ttf' file, " f"got filetype {path_obj.suffix!r} instead")
    from PIL.ImageFont import FreeTypeFont

    try:
        _ = FreeTypeFont(path_obj)
    except OSError as err:
        if path_obj.exists():
            err.add_note(f"{path_obj.resolve()!r}")
        raise err
    src = path_obj.absolute().relative_to(data_root)
    src_pardir = src.parent if not src.is_dir() else src
    if fonts.samefile(src_pardir):
        is_link = False
    else:
        is_link = src.drive == fonts.drive
        path_obj = Path(fonts / (src.stem + ('.ttf' if not is_link else '.lnk')))
    ttf_obj = FreeTypeFont(path_obj)
    ttf_name = ttf_obj.getname()
    if any(ttf_name == FreeTypeFont(v.path).getname() for v in UserFont):
        return print(f"{ttf_name} is already a UserFont member")
    if is_link:
        if path_obj.exists():
            path_obj.unlink()
        path_obj.symlink_to(src)
    else:
        path_obj.write_bytes(src.read_bytes())
    return print(f"Successfully registered new UserFont: {ttf_name}")


def __getattr__(name) -> ...:
    if name.startswith('_'):
        pass
    elif name in _img_data_:
        return partial(Image.open, fp=data_root / Path(_img_data_[name]))
    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")


__all__ = ['UserFont', *_img_data_, 'register_user_font']
