__all__ = ['get_screen_resolution', 'is_vt_proc_enabled']

import ctypes
import os
import subprocess


def get_screen_resolution():
    screen_res: tuple[int, int] | None = None
    if os.name == 'nt':
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        screen_res = (int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1)))
    elif os.name == 'posix':
        from re import findall

        xrandr_out = str(subprocess.Popen(['xrandr'], stdout=subprocess.PIPE).communicate()[0])
        if re_match := findall(r'current\s(\d+) x (\d+)', xrandr_out):
            screen_res = (int(re_match[0][0]), int(re_match[0][1]))
    if screen_res:
        return screen_res
    raise OSError(
        'Unable to find screen resolution. Please use a different operating system')


def is_vt_proc_enabled():
    supports_256 = {'ANSICON', 'COLORTERM', 'ConEmuANSI', 'PYCHARM_HOSTED', 'TERM', 'TERMINAL_EMULATOR',
                    'TERM_PROGRAM', 'WT_SESSION'}
    if not supports_256 & set(os.environ):
        if os.name == 'nt':
            STD_OUTPUT_HANDLE = -11
            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
            if handle == -1:
                return False
            mode = ctypes.c_ulong()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            mode.value |= ENABLE_VIRTUAL_TERMINAL_PROCESSING
            if not kernel32.SetConsoleMode(handle, mode):
                return False
    return True
