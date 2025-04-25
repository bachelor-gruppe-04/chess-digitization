import customtkinter as ctk
import asyncio
import logic.view.state as state
from logic.view.progress_bar_view import ProgressBarTopLevel
from logic.view.reset_specific_board_view import BoardResetSelectorTopLevel

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
  def __init__(self, reset_game_func:any=None, reset_all_games_func:any=None):
    super().__init__()
    self.title("Control Panel")
    self.geometry("800x500")
    self.minsize(600, 400)
    
    self.reset_game_command = reset_game_func
    self.reset_all_games_command = reset_all_games_func
    
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
      command=lambda: self.reset_all_boards()
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
    
    self.bind('<Return>', lambda e: self.apply_number_of_cameras())
    
  def reset_all_boards(self) -> None:
    asyncio.run_coroutine_threadsafe(self._async_reset_all_boards(), state.event_loop)
    
  async def _async_reset_all_boards(self) -> None:
    try:
      await self.reset_all_games_command()
      print("Resetting all boards...")
    except Exception as e:
      import traceback
      print(f"Error resetting all boards: {e}")
      traceback.print_exc()
    
  def validate_entry(self, value:any) -> bool:
    """ Validate the entry to only allow digits and empty string. """
    return value.isdigit() or value == ""
  
  def apply_number_of_cameras(self) -> None:
    """ Apply the number of cameras and start the connection test. """
    number = self.number_of_cameras_entry.get().strip()
    
    if number.isdigit() and int(number) > 0:
      self.number_of_cameras = int(number)
      print(f"Number of cameras set to {self.number_of_cameras}")
      self.disable_main_buttons()
      self.progress_window = ProgressBarTopLevel(self, self.number_of_cameras, self.on_connection_finished)
    else:
      print("Invalid number of cameras. Please enter a positive integer.")
      self.number_of_cameras = 0
      
  def start_tournament(self) -> None:
    """ Start the tournament if cameras are connected. """
    if self.number_of_cameras > 0:
      print("Starting tournament...")
    else:
      print("Please apply a valid number of cameras first.")
      
  def disable_main_buttons(self) -> None:
    """ Disable main buttons during connection test. """
    self.apply_button.configure(state="disabled")
    self.start_button.configure(state="disabled")
    self.reset_select_button.configure(state="disabled")
    self.reset_button.configure(state="disabled")
    self.number_of_cameras_entry.configure(state="disabled")
    
  def enable_main_buttons(self) -> None:
    """ Enable main buttons after connection test. """
    self.apply_button.configure(state="normal")
    self.start_button.configure(state="normal")
    self.reset_select_button.configure(state="normal")
    self.reset_button.configure(state="normal")
    self.number_of_cameras_entry.configure(state="normal")
    
  def on_connection_finished(self, was_cancelled:bool=False) -> None:
    """ Callback when the connection test is finished. """
    if was_cancelled:
      print("Camera test cancelled.")
    else:
      print("Camera test completed.")
      
    self.enable_main_buttons()
    
  def open_board_reset_window(self) -> None:
    """ Open the board reset selector window. """
    if self.number_of_cameras > 0:
      self.disable_main_buttons()
      BoardResetSelectorTopLevel(self, self.number_of_cameras, self.enable_main_buttons, func=self.reset_game_command )
    else:
      print("Please apply a valid number of cameras first.")