import customtkinter as ctk
from progress_bar import ProgressBarTopLevel
from reset_specific_board import BoardResetSelectorTopLevel

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
  def __init__(self):
    super().__init__()
    self.title("Control Panel")
    self.geometry("800x500")
    self.minsize(600, 400)
    
    self.number_of_cameras = 0
    self.progress_window = None
    
    container = ctk.CTkFrame(self, fg_color="transparent")
    container.pack(expand=True)
    
    ctk.CTkLabel(container, text="Control Panel", font=("Segoe UI", 28, "bold")).pack(pady=(10, 10))
    
    vcmd = self.register(self.validate_entry)
    
    self.number_of_cameras_entry = ctk.CTkEntry(
      container,
      placeholder_text="Number of Cameras",
      width=300,
      height=40,
      font=("Segoe UI", 14),
      validate="key",
      validatecommand=(vcmd, '%P')
    )
    self.number_of_cameras_entry.pack(pady=(5, 15))
    
    self.apply_button = ctk.CTkButton(
      container,
      text="Apply Camera Count",
      width=200,
      height=45,
      font=("Segoe UI", 14),
      command=self.apply_number_of_cameras
    )
    self.apply_button.pack(pady=(5, 20))
    
    button_frame = ctk.CTkFrame(container, fg_color="transparent")
    button_frame.pack(pady=10)
    
    self.reset_select_button = ctk.CTkButton(
      button_frame,
      text="Select Which Board to Reset",
      width=200,
      height=45,
      font=("Segoe UI", 14),
      command=self.open_board_reset_window
    )
    self.reset_select_button.pack(side="left", padx=(0, 20))
    
    self.reset_button = ctk.CTkButton(
      button_frame,
      text="Reset All Boards",
      width=200,
      height=45,
      font=("Segoe UI", 14),
      command=lambda: print("Resetting all boards...")
    )
    self.reset_button.pack(side="left")
    
    self.start_button = ctk.CTkButton(
      container,
      text="Start Tournament",
      width=200,
      height=45,
      font=("Segoe UI", 14),
      command=self.start_tournament
    )
    self.start_button.pack(pady=(10, 10))
    
    self.bind('<Return>', lambda: self.apply_number_of_cameras())
    
  def validate_entry(self, value):
    return value.isdigit() or value == ""
  
  def apply_number_of_cameras(self):
    number = self.number_of_cameras_entry.get().strip()
    
    if number.isdigit() and int(number) > 0:
      self.number_of_cameras = int(number)
      print(f"Number of cameras set to {self.number_of_cameras}")
      self.disable_main_buttons()
      self.progress_window = ProgressBarTopLevel(self, self.number_of_cameras, self.on_connection_finished)
    else:
      print("Invalid number of cameras. Please enter a positive integer.")
      self.number_of_cameras = 0
      
  def start_tournament(self):
    if self.number_of_cameras > 0:
      print("Starting tournament...")
    else:
      print("Please apply a valid number of cameras first.")
      
  def disable_main_buttons(self):
    self.apply_button.configure(state="disabled")
    self.start_button.configure(state="disabled")
    self.reset_select_button.configure(state="disabled")
    self.reset_button.configure(state="disabled")
    self.number_of_cameras_entry.configure(state="disabled")
    
  def enable_main_buttons(self):
    self.apply_button.configure(state="normal")
    self.start_button.configure(state="normal")
    self.reset_select_button.configure(state="normal")
    self.reset_button.configure(state="normal")
    self.number_of_cameras_entry.configure(state="normal")
    
  def on_connection_finished(self, was_cancelled=False):
    if was_cancelled:
      print("Camera test cancelled.")
    else:
      print("Camera test completed.")
      
    self.enable_main_buttons()
    
  def open_board_reset_window(self):
    if self.number_of_cameras > 0:
      self.disable_main_buttons()
      BoardResetSelectorTopLevel(self, self.number_of_cameras, self.enable_main_buttons)
    else:
      print("Please apply a valid number of cameras first.")
            
app = App()
app.mainloop()