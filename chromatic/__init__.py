try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"

from . import ascii, color, data
from .ascii import (
    AnsiImage,
    ansi2img,
    ansify,
    ansi_quantize,
    ascii2img,
    ascii_printable,
    contrast_stretch,
    cp437_printable,
    equalize_white_point,
    get_font_key,
    get_font_object,
    img2ansi,
    img2ascii,
    read_ans,
    render_ans,
    reshape_ansi,
    to_sgr_array,
)
from .ascii._glyph_proc import get_glyph_masks
from .color import (
    Back,
    Color,
    ColorNamespace,
    ColorStr,
    Fore,
    SgrParameter,
    Style,
    ansicolor24Bit,
    ansicolor4Bit,
    ansicolor8Bit,
    colorbytes,
    named_color,
)
from .data import register_user_font

__all__ = []

try:
    import os
    import sys
    from functools import lru_cache, wraps
    from types import ModuleType

    def find_modules(path: str):
        from setuptools import find_packages
        from pkgutil import iter_modules

        tree: dict[str, dict | ModuleType] = {
            __name__: {'__module__': sys.modules[__name__]}
        }
        children = set()
        for pkg in find_packages(path):
            children.add(pkg)
            pkg_path = path + '/' + pkg.replace('.', '/')
            if sys.version_info.major == 2 or (
                sys.version_info.major == 3 and sys.version_info.minor < 6
            ):
                for _, name, ispkg in iter_modules([pkg_path]):
                    if not ispkg:
                        children.add(f"{pkg}.{name}")
            else:
                for info in iter_modules([pkg_path]):
                    if not info.ispkg:
                        children.add(f"{pkg}.{info.name}")
        for child in children:
            name = f"{__name__}.{child}"
            depth = tree
            for node in name.split('.'):
                if node not in depth:
                    depth[node] = {}
                depth = depth[node]
            if name in sys.modules:
                depth['__module__'] = sys.modules[name]
        return tree

    def publicize_modules(modulename: str, tree: dict[str, dict | ModuleType]):
        def is_local(obj: object):
            if isinstance(obj, ModuleType):
                return obj.__spec__.parent == modulename
            elif hasattr(obj, '__module__'):
                return obj.__module__ == modulename
            return obj is not None

        for name, subtree in tree.items():
            if name == '__module__' and isinstance(subtree, ModuleType):
                submodule = subtree
                init_dir = dir(submodule)

                @wraps(submodule.__dir__)
                def wrapped():
                    s = set(
                        attr
                        for attr in init_dir
                        if is_local(getattr(submodule, attr, None))
                    )
                    if hasattr(submodule, '__all__'):
                        s.update(submodule.__all__)
                    return list(s)

                sys.modules[submodule.__name__].__dir__ = wrapped

            else:
                publicize_modules(f"{modulename}.{name}", subtree)

    publicize_modules(*find_modules(os.path.split(__file__)[0]).popitem())

finally:
    del find_modules, publicize_modules
