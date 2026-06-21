__all__ = [
    'DEFAULT_FONT',
    'UserFont',
    'VGA437',
    'butterfly',
    'escher',
    'goblin_virus',
    'register_userfont',
    'userfonts',
]

import PIL.Image

from .userfont import DEFAULT_FONT, VGA437, UserFont, register_userfont, userfonts

def butterfly() -> PIL.Image.ImageFile.ImageFile: ...
def escher() -> PIL.Image.ImageFile.ImageFile: ...
def goblin_virus() -> PIL.Image.ImageFile.ImageFile: ...
