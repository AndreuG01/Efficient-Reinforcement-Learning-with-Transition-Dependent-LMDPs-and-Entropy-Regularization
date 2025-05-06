import platform

class TerminalColor:
    ENABLED = None
    CODES = {
        "blue": "\033[34m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "purple": "\033[35m",
        "grey": "\033[90m",
        "orange": "\033[38;5;214m"
    }
    RESET = "\033[0m"

    @staticmethod
    def _init():
        if platform.system() in ["Linux", "Darwin"]:
            TerminalColor.ENABLED = True
        elif platform.system() == "Windows":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
                TerminalColor.ENABLED = True
            except Exception:
                TerminalColor.ENABLED = False

    @staticmethod
    def colorize(text: str, color: str, bold: bool = False) -> str:
        if TerminalColor.ENABLED is None:
            TerminalColor._init()

        if TerminalColor.ENABLED and color in TerminalColor.CODES:
            prefix = "\033[1m" if bold else ""
            return f"{prefix}{TerminalColor.CODES[color]}{text}{TerminalColor.RESET}"
        return text