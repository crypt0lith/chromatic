__all__ = [
    'DEFAULT_FONT',
    'UserFont',
    'VGA437',
    'butterfly',
    'escher',
    'goblin_virus',
    'register_userfont',
    'userfont',
]

import PIL.Image

from .userfont import DEFAULT_FONT, UserFont, VGA437, register_userfont, userfont

def butterfly() -> PIL.Image.ImageFile.ImageFile: ...
def escher() -> PIL.Image.ImageFile.ImageFile: ...
def goblin_virus() -> PIL.Image.ImageFile.ImageFile: ...
