import platform
import re

class TerminalColor:
    """
    A utility class for terminal text coloring and formatting.
    It supports ANSI escape codes for colorizing text in terminals that support it.
    The class automatically detects if the terminal supports ANSI colors and provides methods to colorize text,
    strip ANSI escape codes, and initialize color settings based on the platform.
    """
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
    ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    @staticmethod
    def _init():
        """
        Initializes the terminal color settings based on the platform.
        """
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
        """
        Colorizes the given text with the specified color and optional bold formatting.
        
        Args:
            text (str): The text to colorize.
            color (str): The color to apply. Must be one of the keys in TerminalColor.CODES.
            bold (bool): If True, applies bold formatting to the text.
        
        Returns:
            str: The colorized text with ANSI escape codes, or the original text if coloring is not enabled or the color is invalid.
        """
        if TerminalColor.ENABLED is None:
            TerminalColor._init()

        if TerminalColor.ENABLED and color in TerminalColor.CODES:
            prefix = "\033[1m" if bold else ""
            return f"{prefix}{TerminalColor.CODES[color]}{text}{TerminalColor.RESET}"
        return text

    @staticmethod
    def strip(text: str) -> str:
        """
        Strips ANSI escape codes from the given text.
        
        Args:
            text (str): The text from which to strip ANSI escape codes.
        
        Returns:
            str: The text without ANSI escape codes.
        """
        return TerminalColor.ANSI_ESCAPE_RE.sub("", text)
