import customtkinter as ctk
import asyncio
import logic.view.state as state

class BoardResetSelectorTopLevel(ctk.CTkToplevel):
  def __init__(self, parent, total_cameras:int, on_close_callback, func=None):
    super().__init__(parent)
    self.title("Reset Specific Board")
    self.geometry("400x500")
    self.resizable(False, False)
    self.attributes("-topmost", True)
    self.protocol("WM_DELETE_WINDOW", self.on_close)
    
    self.total = total_cameras
    self.on_close_callback = on_close_callback
    self.function = func

    self.center_on_parent()
    ctk.CTkLabel(self, text="Select Board to Reset", font=("Segoe UI", 20, "bold")).pack(pady=(10, 20))
    
    scroll_frame = ctk.CTkScrollableFrame(self, width=380, height=400)
    scroll_frame.pack(padx=10, pady=10)
    
    for i in range(1, self.total + 1):
      cam_id = i
      cam_label = f"Board {i}"
      
      board_frame = ctk.CTkFrame(scroll_frame, fg_color=("#dddddd", "#2a2a2a"), corner_radius=8)
      board_frame.pack(pady=5, fill="x", padx=5)
      
      ctk.CTkLabel(
        board_frame,
        text=cam_label,
        font=("Segoe UI", 14),
        text_color=("#000000", "#ffffff")
      ).pack(side="left", padx=(10, 0))
      
      reset_button = ctk.CTkButton(
        board_frame,
        text="Reset",
        width=100,
        command=lambda cid=cam_id: self.reset_board(cid)
      )
      reset_button.pack(side="right", padx=(0, 10))
      
  def center_on_parent(self) -> None:
    """ Center the window on the parent window. """
    self.update_idletasks()
    
    px = self.master.winfo_rootx()
    py = self.master.winfo_rooty()
    w = self.master.winfo_width()
    h = self.master.winfo_height()
    
    ww = self.winfo_width()
    wh = self.winfo_height()
    
    x = px + (w - ww) // 2
    y = py + (h - wh) // 2
    
    self.geometry(f"+{x}+{y}")
    
  def reset_board(self, camera_id:int) -> None:
    """ Reset the selected board. """
    asyncio.run_coroutine_threadsafe(self._async_reset_board(camera_id), state.event_loop)
    
  async def _async_reset_board(self, camera_id: int) -> None:
    """ Asynchronous function to reset the selected board. """
    try:
      await self.function(camera_id)
      print(f"Resetting {camera_id}")
    except Exception as e:
      import traceback
      print(f"Error resetting {camera_id}: {e}")
      traceback.print_exc()
    
  def on_close(self) -> None:
    """ Handle the window close event. """
    self.destroy()
    if self.on_close_callback:
      self.on_close_callback()
