import tkinter as tk
import multiprocessing
from multiprocessing.connection import Connection
from typing import List


def rgb(r, g, b):
    """Create an RGB color representation."""
    return f'#{r:02x}{g:02x}{b:02x}'  # Convert RGB values to hexadecimal representation

class FakeInputWindow:
    def __init__(self, message_pipe: Connection):
        self.root = tk.Tk()
        self.root.title("Fake Input Window")

        # check if the Ubuntu Mono font is installed
        self.text_area = tk.Text(self.root, wrap="word", bg=rgb(48, 10, 36), fg="white", font = ("Terminal", 12))
        
        self.text_area.pack(expand=True, fill="both")

        self.text_area.bind("<Key>", self.on_key_press)

        self.message_pipe = message_pipe
        self.waiting_for_input = False

        self.root.after(100, self.control_loop)

        self.root.mainloop()

    def on_key_press(self, event):
        if event.keysym == "Return":
            user_input = self.text_area.get("end-1l", "end-1c")  # Get the last line entered
            if self.waiting_for_input:
                self.message_pipe.send(user_input)
                self.waiting_for_input = False
        
        # Prevent the user from deleting the prompt
        if event.keysym == "BackSpace":
            if self.text_area.get("end-1l", "end-1c") == "":
                return "break"
    

    def control_loop(self):
        if self.message_pipe.poll():
            message_data: List[str, bool] = self.message_pipe.recv()
            message_text: str = message_data[0]
            is_input: bool = message_data[1]
            
            if is_input:
                self.waiting_for_input = True
                self.print(message_text)
            
            else:
                self.print(message_text)
        
        self.root.after(100, self.control_loop)

    def print(self, message: str):
        self.text_area.insert(tk.END, f"{message}\n")
        self.text_area.see(tk.END)  # Auto-scroll to the bottom

    
class UserInput:
    def __init__(self):
        self.message_pipe, child_pipe = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=self.get_user_input, args=(child_pipe,))
        self.process.start()

    def input(self, message: str) -> str:
        self.message_pipe.send([message, True])
        return self.message_pipe.recv()
    
    def print(self, message: str):
        self.message_pipe.send([message, False])

    @staticmethod
    def get_user_input(message_pipe: Connection):
        FakeInputWindow(message_pipe)
        
    # close the process
    def close(self):
        self.process.terminate()

    # close the process when the object is deleted
    def __del__(self):
        self.close()

    # allow the object to be used as a context manager (with statement)
    def __enter__(self):
        return self

    # close the process when the context manager is exited (exit with statement)
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == '__main__':
    with UserInput() as user_input:
        user_input.print("Hello world!")
        print(user_input.input("What is your name? "))
        print(user_input.input("What is your age? "))