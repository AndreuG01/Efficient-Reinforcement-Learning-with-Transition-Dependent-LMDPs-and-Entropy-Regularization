class CustomPalette:
    ROYAL_PURPLE = "#640D5F"
    HOT_PINK = "#D91656"
    FIERY_ORANGE = "#EB5B00"
    GOLDEN_YELLOW = "#FFB200"
    OCEAN_BLUE = "#2D728F"
    FRESH_GREEN = "#3E8914"
    NEON_MAGENTA = "#FF00D3"
    ELECTRIC_GREEN = "#32FF00"
    RICH_BROWN = "#743500"
    MINTY_BLUE = "#7DFFFF"
    LEMON_YELLOW = "#FBFF00"
    VIBRANT_PURPLE = "#C300FF"
    TROPICAL_TEAL = "#00CDAA"
    PEACHY_APRICOT = "#FFA87E"
    DARK_BURGUNDY = "#331E23"
    BRIGHT_RED = "#FF0000"
    COBALT_BLUE = "#0047AB"
    BLUE_VIOLET = "#8A2BE2"
    GOLD = "#FFD700"
    FOREST_GREEN = "#228B22"
    
    def __init__(self):
        self.__colors = [
            value for key, value in vars(self.__class__).items()
            if isinstance(value, str) and value.startswith("#")
        ]
    
    
    def __len__(self):
        return len(self.__colors)
    
    def __getitem__(self, index):
        return self.__colors[index]

