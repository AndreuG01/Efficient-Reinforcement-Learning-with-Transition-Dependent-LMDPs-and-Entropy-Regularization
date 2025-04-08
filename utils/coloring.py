import platform

#TODO: test in windows
class TerminalColor:
    ENABLED = False
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
    def init():
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
    def colorize(text: str, color: str) -> str:
        if TerminalColor.ENABLED and color in TerminalColor.CODES:
            return f"{TerminalColor.CODES[color]}{text}{TerminalColor.RESET}"
        return text