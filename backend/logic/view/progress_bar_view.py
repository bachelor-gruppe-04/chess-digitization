import customtkinter as ctk

class ProgressBarTopLevel(ctk.CTkToplevel):
  """ A progress bar window that shows the progress of connecting to cameras. """
  def __init__(self, parent, total_cameras:int, on_finish_callback):
    super().__init__(parent)
    self.title("Connecting to Cameras...")
    self.geometry("420x130")
    self.resizable(False, False)
    self.attributes("-topmost", True)
    
    self.parent = parent
    self.total = total_cameras
    self.current = 0
    self.on_finish_callback = on_finish_callback
    self.cancelled = False
    
    self.center_on_parent()
    self.protocol("WM_DELETE_WINDOW", self.cancel_connection)
    
    self.progressbar = ctk.CTkProgressBar(self, width=320, height=20)
    self.progressbar.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10))
    self.progressbar.set(0)
    
    cancel_button = ctk.CTkButton(
        self,
        text="Cancel",
        width=100,
        command=self.cancel_connection
    )
    cancel_button.grid(row=1, column=1, sticky="e", padx=20, pady=(5, 10))
    
    self.fade_in()
    self.update_progress()
    
  def center_on_parent(self) -> None:
    """ Center the window on the parent window. """
    self.update_idletasks()
    
    px = self.parent.winfo_rootx()
    py = self.parent.winfo_rooty()
    w = self.parent.winfo_width()
    h = self.parent.winfo_height()
    
    ww = self.winfo_width()
    wh = self.winfo_height()
    
    x = px + (w - ww) // 2
    y = py + (h - wh) // 2
    
    self.geometry(f"+{x}+{y}")
    
  def fade_in(self, alpha:float=0.0) -> None:
    """ Fade in the window. """
    if alpha < 1.0:
      self.attributes("-alpha", alpha)
      self.after(30, lambda: self.fade_in(alpha + 0.1))
    else:
      self.attributes("-alpha", 1.0)
      
  def update_progress(self) -> None:
    """ Update the progress bar. """
    if self.cancelled:
      return None
    
    if self.current <= self.total:
      progress = self.current / self.total
      self.progressbar.set(progress)
      self.current += 1
      self.after(300, self.update_progress)
    else:
      self.after(700, self.finish_connection)
      
  def cancel_connection(self) -> None:
    """ Cancel the connection. """
    self.cancelled = True
    self.destroy()
    self.on_finish_callback()
    
  def finish_connection(self) -> None:
    """ Finish the connection. """
    self.destroy()
    self.on_finish_callback()