import threading
import time
import itertools


class Spinner:
    """
    A simple spinner class to show a loading animation in the console.
    This class uses a separate thread to animate the spinner while a task is running.
    Attributes:
        message (str): The message to display alongside the spinner.
        padding (int): The number of characters to pad the message.
        spinner (itertools.cycle): An iterator that cycles through spinner characters.
        running (bool): A flag to indicate if the spinner is running.
        _thread (threading.Thread): The thread that runs the spinner animation.
    """
    def __init__(self, message: str, padding: int = 50):
        """
        Initializes the Spinner with a message and padding.
        Args:
            message (str): The message to display alongside the spinner.
            padding (int): The number of characters to pad the message.
        """
        self.message = message
        self._spinner_characters = ["|", "/", "-", "\\"]
        self.spinner = itertools.cycle(self._spinner_characters)
        self.running = False
        self._thread = None
        self.padding = padding
        self.time = None
    
    
    def __animate(self):
        """
        The method that runs in a separate thread to animate the spinner.
        It continuously updates the spinner character and prints the message until stopped.
        The spinner character is cycled through a set of characters to create the animation effect.
        The message is printed with a specified padding to ensure proper alignment.
        The method uses a while loop to keep running until the `running` flag is set to False.
        
        Args:
            None
        
        Returns:
            None
        """
        while self.running:
            print(f"{self.message} {next(self.spinner)}".ljust(self.padding), end="\r")
            time.sleep(0.1)
        
    def start(self):
        """
        Starts the spinner animation by setting the `running` flag to True and starting the animation thread.
        The method creates a new thread that runs the `__animate` method, which handles the spinner animation.
        
        Args:
            None
        
        Returns:
            None
        """
        self.running = True
        self._thread = threading.Thread(target=self.__animate)
        self.time = time.time()
        self._thread.start()
    
    def stop(self, interrupted: bool = False):
        """
        Stops the spinner animation by setting the `running` flag to False and joining the animation thread.
        The method waits for the animation thread to finish before proceeding.
        This ensures that the spinner stops cleanly and does not leave any hanging threads.
        
        Args:
            interrupted (bool): A flag to indicate if the spinner was interrupted. If True, it will not print the finished message.
        
        Returns:
            None
        """
        self.time = time.time() - self.time
        self.running = False
        self._thread.join()
        if not interrupted:
            print(f"{self.message}\tFINISHED in {round(self.time, 3)}s")