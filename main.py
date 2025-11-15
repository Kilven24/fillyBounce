import configparser
import cv2
import os
import queue
import threading
import streamlink
import time
import tkinter as tk
import torch
from collections import deque
from tkinter import ttk, simpledialog, filedialog
from ultralytics import YOLO


# Configuration constants
DEFAULT_CONFIG = {
    'twitch_channel_url': 'https://www.twitch.tv/Filian',
    'performance_model_path': 'Models/trained_n_int8_openvino_model',
    'precision_model_path': 'Models/trained_m.engine',
    'use_performance_model': 'False',
    'model_confidence': '0.8',
    'tk_showframe': 'True',
    'tk_model_verbose': 'False',
    'tk_save_lowscores': 'False',
    'relative_jump_threshold': '2',
    'obs_device': '0',
    'current_time': '0',
    'counter': '0',
    'oauth_token': 'YOUR-OAUTH-TOKEN-DO-NOT-SHARE',
    'dev_test': 'False'
}

REQUIRED_CONFIG_VARS = [
    "twitch_channel_url", "performance_model_path", "precision_model_path", 'use_performance_model', "model_confidence",
    "tk_showframe", "tk_model_verbose", "tk_save_lowscores", "relative_jump_threshold",
    "obs_device", "current_time", "counter", "oauth_token", "dev_test"
]

CONFIG_FILE = 'config.ini'

def select_file():
    """Open file dialog and return the selected file path."""
    file_path = filedialog.askopenfilename()
    if file_path:
        print(f"Selected file: {file_path}")
    else:
        print("No file selected.")
    return file_path

def promptuser(prompt_message):
    """Display input dialog and return user input."""
    root = tk._default_root
    if root:
        # Find the active toplevel window (could be options window)
        parent = root.focus_get()
        if parent:
            parent = parent.winfo_toplevel()
        else:
            parent = root
        
        # Create dialog
        dialog = tk.Toplevel(parent)
        dialog.withdraw()  # Hide while positioning
        dialog.title("Input")
        
        # Create input field
        tk.Label(dialog, text=prompt_message, padx=20, pady=10).pack()
        entry = tk.Entry(dialog, width=40)
        entry.pack(padx=20, pady=10)
        entry.focus_set()
        
        result = [None]
        
        def on_ok():
            result[0] = entry.get()
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        # Buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        tk.Button(button_frame, text="OK", command=on_ok, width=10).pack(side="left", padx=5)
        tk.Button(button_frame, text="Cancel", command=on_cancel, width=10).pack(side="left", padx=5)
        
        # Bind Enter and Escape keys
        entry.bind('<Return>', lambda e: on_ok())
        dialog.bind('<Escape>', lambda e: on_cancel())
        
        # Update to get proper size
        dialog.update_idletasks()
        
        # Position relative to parent window
        parent_x = parent.winfo_x()
        parent_y = parent.winfo_y()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()
        
        dialog_width = dialog.winfo_width()
        dialog_height = dialog.winfo_height()
        
        # Center on parent window
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        dialog.geometry(f"+{x}+{y}")
        dialog.deiconify()  # Show dialog
        
        # Make modal
        dialog.transient(parent)
        dialog.grab_set()
        parent.wait_window(dialog)
        
        return result[0]
    else:
        # Fallback to simpledialog if no root exists
        return simpledialog.askstring("Input", prompt_message)

def check_config():
    """Verify that all required configuration variables exist."""
    config = configparser.ConfigParser()
    
    try:
        config.read(CONFIG_FILE)
        
        if 'DEFAULT' not in config:
            print("Invalid config file structure. Creating default config.")
            default_config()
            return
        
        missing_vars = [var for var in REQUIRED_CONFIG_VARS if var not in config['DEFAULT']]
        
        if missing_vars:
            print(f"Missing variables in config.ini: {', '.join(missing_vars)}")
            print("Creating default config file.")
            default_config()
            return
        
        print("Config.ini validated successfully. All required variables present.")
        
    except Exception as e:
        print(f"Error reading config file: {e}")
        print("Creating default config file.")
        default_config()

def default_config():
    """Create a default configuration file."""
    try:
        config = configparser.ConfigParser()
        config['DEFAULT'] = DEFAULT_CONFIG
        
        with open(CONFIG_FILE, 'w') as configfile:
            config.write(configfile)
        
        print(f"Default config file created: {CONFIG_FILE}")
        
    except Exception as e:
        print(f"Error creating default config file: {e}")
        raise

def load_config():
    """Load configuration file, creating default if it doesn't exist."""
    if not os.path.isfile(CONFIG_FILE):
        print(f"Config file '{CONFIG_FILE}' not found. Creating default config file.")
        default_config()
    else:
        print("Config file found.")
        check_config()
    
    try:
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE)
        
        # Verify config loaded successfully
        if 'DEFAULT' not in config or not config['DEFAULT']:
            print("Failed to load config. Creating default config.")
            default_config()
            config.read(CONFIG_FILE)
        
        return config
        
    except Exception as e:
        print(f"Error loading config file: {e}")
        print("Attempting to create default config.")
        default_config()
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE)
        return config

class MyApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Load configuration
        config = load_config()
        
        # Initialize window
        self.setup_window()
        
        # Initialize configuration variables
        self.load_config_variables(config)
        
        # Initialize runtime variables
        self.initialize_runtime_variables()
        #check if gpu is usable for pytorch
        self.gpu_check()
        #check if model_path exists
        self.model_path_check()
        # Setup UI
        #self.setup_background()
        self.setup_widgets()
        # Start recurring tasks
        self.iterate_time()
        self.update_label_counter()
        self.jumps_per_second()
        # Task queue for thread management
        self.task_queue = queue.Queue()
        self.check_queue()

        # Handle window close event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    # Parent window
    def on_closing(self):
        """Handle application shutdown gracefully."""
        print("Closing application...")
        self.frameloop = False
        self.stop_timer()
        cv2.destroyAllWindows()
        self.destroy()

    def setup_window(self):
        """Configure main window properties."""
        self.options_window = None
        self.title("fillyBounce")
        self.geometry("350x550")
        self.configure(bg='#2D1C3A')
        self.iconbitmap('icon.ico')

        # Prevent window resizing
        self.resizable(False, False)

        # Configure styles
        # Color Pallet
        # background='#69657A', foreground='#E3CECF'
        # Active - background=[('active', '#C4703D')], foreground=[('active', '#E3CECF')])

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure('TButton', background='#69657A', foreground='#E3CECF')
        self.style.map('TButton',
                       background=[('active', '#C4703D')],
                       foreground=[('active', '#E3CECF')])

        self.style.configure('TLabel', background='#69657A', foreground='#E3CECF',
                           relief="solid", highlightthickness=5)
        self.style.configure('Custom.TLabel', background='#69657A', foreground='#e89c7b',)

        self.style.configure('TCheckbutton', background='#69657A', foreground='#E3CECF')
        self.style.map("TCheckbutton",
                  background=[('active', '#C4703D')],
                  foreground=[('active', '#E3CECF')])

    def load_config_variables(self, config):
        """Load configuration variables from config file."""
        self.twitch_channel_url = config['DEFAULT']['twitch_channel_url']
        self.performance_model_path = config['DEFAULT']['performance_model_path']
        self.precision_model_path = config['DEFAULT']['precision_model_path']
        self.use_performance_model = tk.BooleanVar(self, value=config['DEFAULT'].getboolean('use_performance_model'))
        self.model_confidence = config['DEFAULT'].getfloat('model_confidence')
        self.tk_showframe = tk.BooleanVar(self, value=config['DEFAULT'].getboolean('tk_showframe'))
        self.tk_model_verbose = tk.BooleanVar(self, value=config['DEFAULT'].getboolean('tk_model_verbose'))
        self.tk_save_lowscores = tk.BooleanVar(self, value=config['DEFAULT'].getboolean('tk_save_lowscores'))
        self.relative_jump_threshold = config['DEFAULT'].getfloat('relative_jump_threshold')
        self.obs_device = config['DEFAULT'].getint('obs_device')
        self.current_time = config['DEFAULT'].getint('current_time')
        self.counter = config['DEFAULT'].getint('counter')
        self.oauth_token = config['DEFAULT']['oauth_token']
        self.dev_test = config['DEFAULT'].getboolean('dev_test')

    def initialize_runtime_variables(self):
        """Initialize runtime variables."""
        self.tk_counter = tk.StringVar(self, value=0)
        self.tk_jps = tk.DoubleVar(self, value=0.0)
        self.counter_trigger = False
        self.options_window_open = False
        self.timer_running = False
        self.ypos = deque(maxlen=10)
        self.xypos = deque(maxlen=10)

    def setup_widgets(self):
        """Create and configure all UI widgets."""
        # Main capture buttons
        self.twitch_cap_btn = ttk.Button(self, text="Twitch Capture",command=lambda: self.start_task(3))
        self.twitch_cap_btn.pack(pady=5)

        self.processvideo_btn = ttk.Button(self, text="Process Recorded Video",command=lambda: self.start_task(1))
        self.processvideo_btn.pack()

        self.obs_cap_btn = ttk.Button(self, text="OBS/Cam Capture",command=lambda: self.start_task(2))
        self.obs_cap_btn.pack(pady=5)

        ttk.Button(self, text="Stop Capture", command=self.stop_tasks).pack()

        # Counter display
        self.setup_counter_display()

        # jps display
        self.setup_jps_display()

        # Timer display
        self.time_label = ttk.Label(self, text="00:00:00", font=("Arial", 48), style="Custom.TLabel")
        self.time_label.pack(pady=10)

        # Control buttons
        self.setup_control_buttons()

    def setup_counter_display(self):
        """Set up counter label and display."""
        self.counter_description = ttk.Label(self, text="Jump Counter")
        self.counter_description.pack(pady=5)
        self.counter_description.config(font=("Arial", 16, "bold"))

        self.counter_label = ttk.Label(self, textvariable=self.tk_counter, style="Custom.TLabel")
        self.counter_label.pack(pady=5)
        self.counter_label.config(font=("Arial", 48, "bold"))

    def setup_jps_display(self):
        """Setup jumps per second display."""
        self.tk_jps_description = ttk.Label(self, text="Jumps per second")
        self.tk_jps_description.pack(pady=5)
        self.tk_jps_description.config(font=("Arial", 16, "bold"))

        self.tk_jps_label = ttk.Label(self, textvariable=self.tk_jps, style="Custom.TLabel")
        self.tk_jps_label.pack(pady=5)
        self.tk_jps_label.config(font=("Arial", 16, "bold"))

    def setup_control_buttons(self):
        """Setup counter control and timer buttons."""
        # Counter buttons
        self.counter_btn_frame = ttk.Frame(self)
        self.counter_btn_frame.pack()

        ttk.Button(
            self.counter_btn_frame, text="+1",
            command=lambda: self.delta_counter(1)
        ).pack(side="left")

        self.counter_accept_Button = ttk.Button(
            self.counter_btn_frame, text="Set Counter",
            command=self.set_counter
        )
        self.counter_accept_Button.pack(side="left")

        ttk.Button(
            self.counter_btn_frame, text="-1",
            command=lambda: self.delta_counter(-1)
        ).pack(side="left")

        ttk.Button(self, text="Set Time", command=self.prompt_user_time).pack()

        # Options button
        self.options_btn = ttk.Button(self, text="Options", command=self.options_menu)
        self.options_btn.pack(side="bottom", anchor="sw")

    # Counter
    def update_label_counter(self):
        """Update the counter label with the current value."""
        try:
            formated_counter = "{:,}".format(self.counter)
            self.tk_counter.set(formated_counter)
        except (ValueError, TypeError) as e:
            print(f"Error updating counter display: {e}")
            self.tk_counter.set(0)
        finally:
            self.counter_label.after(100, self.update_label_counter)

    def delta_counter(self, value):
        """Increase or decrease the counter by the specified value with validation."""
        try:
            delta = int(value)
            new_value = self.counter + delta

            # Prevent negative counter values
            if new_value < 0:
                print(f"Counter cannot be negative. Current value: {self.counter}")
                return

            self.counter = new_value

        except (ValueError, TypeError) as e:
            print(f"Error adjusting counter: {e}")

    def set_counter(self):
        """Set the counter to a user-specified value with validation."""
        user_input = promptuser("Enter the desired counter value:")

        if not user_input:
            print("Setting counter to 0...")
            self.counter = 0
            return

        try:
            new_value = int(user_input)

            if new_value < 0:
                print(f"Invalid counter value: {new_value}. Must be non-negative.")
                return

            self.counter = new_value
            print(f"Counter set to: {self.counter}")

        except ValueError:
            print(f"Invalid input: '{user_input}'. Please enter an integer value.")

    # Options
    def options_menu(self):
        """Create and display the options window."""
        if self.options_window_open:
            return

        self.options_window_open = True
        self.options_window = tk.Toplevel(self)
        self.options_window.title("Options")
        self.options_window.geometry("300x200")
        self.options_window.configure(bg='#2D1C3A')

        # Checkbuttons

        ttk.Checkbutton(self.options_window, text="Show Frame", variable=self.tk_showframe).pack(side="top")
        ttk.Checkbutton(self.options_window, text="Use Performance Model",
        variable=self.use_performance_model).pack(side="top")

        if self.dev_test:
            ttk.Checkbutton(self.options_window, text="Model Verbose", variable=self.tk_model_verbose).pack(side="top")

            ttk.Checkbutton(self.options_window, text="Save Low Scores",
            variable=self.tk_save_lowscores).pack(side="top")

        # Configuration buttons
        ttk.Button(self.options_window, text="Change Twitch Channel url",
        command=self.set_twitch_channel_url).pack(side="top")

        ttk.Button(self.options_window, text="Set obs Device Number", command=self.set_obs).pack(side="top")

        if self.dev_test:
            ttk.Button(self.options_window, text="Relative jump threshold",command=self.set_relative_jump_threshold
            ).pack(side="top")

            ttk.Button(self.options_window, text="Model confidence", command=self.set_model_confidence
            ).pack(side="top")

        ttk.Button(self.options_window, text="Close", command=self.close_options).pack(side="bottom")

        # Update to get proper size
        self.options_window.update_idletasks()

        # Position relative to main window
        main_x = self.winfo_x()
        main_y = self.winfo_y()
        main_width = self.winfo_width()
        main_height = self.winfo_height()

        options_width = self.options_window.winfo_width()
        options_height = self.options_window.winfo_height()

        # Center on main window
        x = main_x + (main_width - options_width) // 2
        y = main_y + (main_height - options_height) // 2

        self.options_window.geometry(f"300x200+{x}+{y}")

        # Bind to window destroy event to ensure flag is reset
        self.options_window.bind("<Destroy>", self.on_options_destroy)
        self.options_window.protocol("WM_DELETE_WINDOW", self.close_options)

    def on_options_destroy(self, event):
        """Handle options window destruction to reset flag."""
        # Only reset if the destroyed widget is the options window itself
        if event.widget == self.options_window:
            self.options_window_open = False

    def close_options(self):
        """Close options window and save settings to the config file."""
        try:
            self.save_config_settings()
            print("Settings saved successfully.")
        except Exception as e:
            print(f"Error saving settings: {e}")
        finally:
            if self.options_window:
                self.options_window.destroy()

    def save_config_settings(self):
        """Save current settings to the config file."""
        config = configparser.ConfigParser()
        config.read('config.ini')

        # Define settings to save
        settings = {
            'use_performance_model': str(self.use_performance_model.get()),
            'tk_showframe': str(self.tk_showframe.get()),
            'tk_model_verbose': str(self.tk_model_verbose.get()),
            'tk_save_lowscores': str(self.tk_save_lowscores.get()),
            'twitch_channel_url': str(self.twitch_channel_url),
            'obs_device': str(self.obs_device),
            'relative_jump_threshold': str(self.relative_jump_threshold),
            'model_confidence': str(self.model_confidence)
        }

        # Update config with settings
        for key, value in settings.items():
            config.set('DEFAULT', key, value)

        # Write to file
        with open('config.ini', 'w') as configfile:
            config.write(configfile)

    def set_obs(self):
        """Set OBS device number based on user input with validation."""
        user_input = promptuser("Set OBS device number (e.g., 0, 1, 2):")

        if not user_input:
            print("OBS device number unchanged.")
            return

        try:
            device_num = int(user_input)

            if device_num < 0:
                print(f"Invalid device number: {device_num}. Must be non-negative.")
                return

            self.obs_device = device_num
            print(f"OBS device number set to: {self.obs_device}")

        except ValueError:
            print(f"Invalid input: '{user_input}'. Please enter an integer value.")

    def set_twitch_channel_url(self):
        """Set Twitch URL based on user input with validation."""
        user_input = promptuser("Set Twitch channel URL (e.g., https://www.twitch.tv/CHANNEL):")

        if not user_input:
            print("Twitch channel URL unchanged.")
            return

        # Strip whitespace
        url = user_input.strip()

        # Basic validation - check if it looks like a Twitch URL
        if not url.startswith(('https://www.twitch.tv/', 'https://twitch.tv/')):
            print(f"Warning: URL may not be a valid Twitch channel URL: {url}")
            print("Expected format: https://www.twitch.tv/CHANNEL")
            # Still allow it but warn the user

        self.twitch_channel_url = url
        print(f"Twitch channel URL set to: {self.twitch_channel_url}")

    def quality_check(self, streams):
        """Select optimal stream quality from available streams."""
        for selected_quality in reversed(streams):
            print(selected_quality)
            if selected_quality in ("720p60", "480p", "480p30"):
                return selected_quality

        # Preferred qualities in order of preference
        preferred_qualities = ("480p", "480p30", "480p60", "360p", "360p30")

        print("Checking available stream qualities...")

        # Check each available quality (in reverse order for higher qualities first)
        for quality in reversed(list(streams.keys())):
            print(f"  Available: {quality}")

            if quality in preferred_qualities:
                print(f"Selected quality: {quality}")
                return quality

        print("No preferred quality found in available streams.")
        return None

    def set_relative_jump_threshold(self):
        """Set relative jump threshold based on user input with validation."""
        user_input = promptuser("Set relative jump threshold (e.g., 2 to 2.5. Lower numbers require higher jumps to count) :")

        if not user_input:
            print("Relative jump threshold unchanged.")
            return

        try:
            threshold = float(user_input)

            if threshold <= 0:
                print(f"Invalid threshold value: {threshold}. Must be greater than 0")
                return

            self.relative_jump_threshold = threshold
            print(f"Relative jump threshold set to: {self.relative_jump_threshold}")

        except ValueError:
            print(f"Invalid input: '{user_input}'. Please enter a numeric value.")

    def set_model_confidence(self):
        """Set model confidence based on user input with validation."""
        user_input = promptuser("Set model confidence (0.0 - 1.0):")

        if not user_input:
            print("Model confidence unchanged.")
            return

        try:
            confidence = float(user_input)

            if not 0.0 <= confidence <= 1.0:
                print(f"Invalid confidence value: {confidence}. Must be between 0.0 and 1.0")
                return

            self.model_confidence = confidence
            print(f"Model confidence set to: {self.model_confidence}")

        except ValueError:
            print(f"Invalid input: '{user_input}'. Please enter a numeric value.")

    def gpu_check(self):
        """Check if GPU is available and set appropriate model path."""

        print("Checking if GPU is available...")
        if not torch.cuda.is_available():
            print("CUDA is not available. Switching to CPU")
            print("setting performance model active. Do not change as other model will crash if cpu used")
            self.use_performance_model = True

            self.hardware = "cpu"
            return

        print("CUDA is available!")

        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU name: {torch.cuda.get_device_name(0)}")

        self.hardware = "cuda"

    def model_path_check(self):
        """Check if model path exists and exit if not found."""
        print("Checking if model is available...")
        if self.hardware == 'cuda' and not os.path.exists(self.precision_model_path):
            print(f"Model path does not exist: {self.precision_model_path}")
            print("Exiting. Please download the model from the GitHub repository.")
            exit('Model Missing')

        if self.hardware == 'cpu' and not os.path.exists(self.performance_model_path):
            print("Model is not available. Please download the model from the GitHub repository.")
            print(f"Expected model path: {self.performance_model_path}")
            exit('Model Missing')

        self.model_path = self.precision_model_path
        print("Model is available!")
        #print(self.model_path)

    # Task Management
    def start_task(self, tasknum):
        """Start a task in a separate thread and disable the corresponding button."""
        task_config = {}
        task_config[1] = (self.processvideo_btn, self.scanning_processvideo)
        task_config[2] = (self.obs_cap_btn, self.scanning_obs)
        task_config[3] = (self.twitch_cap_btn, self.scanning_twitch)

        button, target = task_config.get(tasknum)
        button.config(text="Running", state=tk.DISABLED)
        threading.Thread(target=target, daemon=True).start()

    def stop_tasks(self):
        """Stop all running tasks and restore buttons."""
        print("Stopping tasks...")
        self.restore_button_states()
        self.stop_timer()
        self.frameloop = False

    def restore_button_states(self):
        """Restore all capture buttons to their initial state."""
        button_configs = [
            (self.twitch_cap_btn, "Twitch Capture"),
            (self.processvideo_btn, "Process Recorded Video"),
            (self.obs_cap_btn, "OBS/Cam Capture")
        ]

        for button, text in button_configs:
            button.config(text=text, state=tk.NORMAL)

    def check_queue(self):
        """Check task queue and update button states when tasks complete."""
        try:
            while True:
                message = self.task_queue.get_nowait()

                if message == "Task1":
                    self.processvideo_btn.config(text="Process Recorded Video", state=tk.NORMAL)
                elif message == "Task2":
                    self.obs_cap_btn.config(text="OBS/Cam Capture", state=tk.NORMAL)
                elif message == "Task3":
                    self.twitch_cap_btn.config(text="Twitch Capture", state=tk.NORMAL)
        except queue.Empty:
            pass

        self.after(100, self.check_queue)

    # Scanning Choices
    def scanning_processvideo(self):
        """Set up video capture from a video file."""
        try:
            # Prompt user to select a video file
            video_path = select_file()

            # Check if user cancelled file selection
            if video_path is None:
                print("No file selected. Returning to main menu.")

                self.processvideo_btn.config(text="Process Recorded Video", state=tk.NORMAL)
                return

            # Validate that the file exists
            if not os.path.exists(video_path):
                print(f"Error: File not found: {video_path}")
                self.processvideo_btn.config(text="Process Recorded Video", state=tk.NORMAL)
                return

            # Validate file extension
            valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')
            if not video_path.lower().endswith(valid_extensions):
                print(f"Warning: File may not be a valid video format: {video_path}")

            print(f"Processing video: {video_path}")

            # Start processing the video
            self.scanning(video_path, queueref="Task1", video_path=video_path)

        except Exception as e:
            print(f"Error in scanning_processvideo: {e}")
            self.processvideo_btn.config(text="Process Recorded Video", state=tk.NORMAL)

    def scanning_obs(self):
        """Setup video capture for OBS virtual camera."""
        cap = None
        try:
            print(f"Attempting to open Cam (device {self.obs_device})...")

            # Try to open the OBS device
            cap = cv2.VideoCapture(self.obs_device)

            # Check if the device opened successfully
            if not cap.isOpened():
                print(f"Error: Unable to open Cam device {self.obs_device}")
                print("Possible causes:")
                print("  - Cam Device is not running")
                print("  - Incorrect device number in settings")
                print("  - Device is being used by another application")
                self.obs_cap_btn.config(text="OBS/Cam Capture", state=tk.NORMAL)
                return

            # Test if we can actually read a frame
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                print("Error: Cam device opened but cannot read frames")
                cap.release()
                self.obs_cap_btn.config(text="OBS/Cam Capture", state=tk.NORMAL)
                return

            print("Cam Device opened successfully")
            cap.release()

            # Start processing the OBS stream
            self.scanning(self.obs_device, queueref="Task2")

        except Exception as e:
            print(f"Error in scanning_obs: {e}")
            self.obs_cap_btn.config(text="OBS/Cam Capture", state=tk.NORMAL)
        finally:
            # Ensure capture is released if still open
            if cap is not None and cap.isOpened():
                cap.release()

    def scanning_twitch(self):
        """Prepare and start Twitch stream capture."""
        try:
            # Create and configure Streamlink session
            print(f"Connecting to Twitch channel: {self.twitch_channel_url}")
            session = streamlink.Streamlink()

            #Token check
            if self.oauth_token == 'YOUR-OAUTH-TOKEN-DO-NOT-SHARE':
                print("No OAuth token found")
                session_options = {
                    'low-latency': True,
                    'stream-timeout': 30,
                    'twitch-disable-ads': True
                }
            else:
                print("Using OAuth token")
                session_options = {
                    'http-headers': {'Authorization': f'OAuth {self.oauth_token}'},
                    'low-latency': True,
                    'stream-timeout': 30,
                    'twitch-disable-ads': True
                }
            for option, value in session_options.items():
                session.set_option(option, value)

            # Fetch available streams
            streams = session.streams(url=self.twitch_channel_url)

            if not streams:
                print("No streams found for the URL. The channel may be offline.")
                self.twitch_cap_btn.config(text="Twitch Capture", state=tk.NORMAL)
                return

            # Select optimal quality
            quality_selected = self.quality_check(streams)

            if quality_selected is None:
                print("Optimal quality (480p/480p30) not found.")
                print(f"Available qualities: {list(streams.keys())}")
                self.twitch_cap_btn.config(text="Twitch Capture", state=tk.NORMAL)
                return

            # Get stream URL and start processing
            quality_stream = streams[quality_selected]
            play_url = quality_stream.url

            print(f"Selected quality: {quality_selected}")
            print(f"Starting Twitch capture...")

            self.scanning(play_url, queueref="Task3")

        except Exception as e:
            print(f"Error setting up Twitch capture: {e}")
            self.twitch_cap_btn.config(text="Twitch Capture", state=tk.NORMAL)

    # Frame Processing
    def scanning(self, source, queueref, video_path=None):
        """Main scanning function that coordinates all processing threads."""
        grabber_thread = None
        processor_thread = None
        writer_thread = None
        try:
            # Initialize processing
            print("Initializing video processing...")
            self.frameloop = True
            self.start_timer()

            # Load YOLO model
            if self.use_performance_model.get():
                self.model_path = self.performance_model_path
                print("Using performance model")
            else:
                self.model_path = self.precision_model_path
                print("Using precision model")
            print(f"Loading model from: {self.model_path}")
            model = YOLO(self.model_path)

            # CPU-specific optimizations
            if self.hardware == "cpu":

                # Warm up model
                print("Warming up model...")
                dummy = torch.zeros((1, 3, 416, 416))
                _ = model(dummy, verbose=False)
                print("Model ready")

            # Create queues for thread communication
            frame_queue = queue.Queue(maxsize=60)
            result_queue = queue.Queue(maxsize=60)
            write_queue = queue.Queue(maxsize=60) if video_path else None

            # Start frame grabber thread
            print("Starting frame grabber thread...")
            grabber_thread = threading.Thread(
                target=self.frame_grabber,
                args=(source, frame_queue),
                daemon=True,
                name="FrameGrabber"
            )
            grabber_thread.start()

            # Start detection processor thread
            print("Starting detection processor thread...")
            processor_thread = threading.Thread(
                target=self.detection_processor,
                args=(model, frame_queue, result_queue, write_queue, video_path),
                daemon=True,
                name="DetectionProcessor"
            )
            processor_thread.start()

            # Start video writer thread if video_path is provided
            if video_path is not None:
                print("Starting video writer thread...")
                writer_thread = threading.Thread(
                    target=self.frame_writer,
                    args=(write_queue, video_path),
                    daemon=True,
                    name="FrameWriter"
                )
                writer_thread.start()

            print("All threads started successfully. Processing frames...")

            # Main display loop
            self.display_frames(result_queue)

        except Exception as e:
            print(f"Error in scanning: {e}")
            self.frameloop = False

        finally:
            # Cleanup
            print("Cleaning up resources...")
            self.cleanup_scanning(grabber_thread, processor_thread, writer_thread, queueref)

    def display_frames(self, result_queue):
        """Display processed frames in a window."""
        try:
            while self.frameloop:
                try:
                    # Get frame with timeout to prevent blocking
                    frame = result_queue.get(timeout=1)

                    # Display frame if show frame option is enabled
                    if self.tk_showframe.get():
                        cv2.imshow('Processing', frame)
                    fps = self.writerfps
                    fps_to_ms = int((1.0 / fps) * 1000)
                    # Check for 'q' key press to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("User pressed 'q' to quit")
                        self.frameloop = False
                        break

                    # time.sleep()

                except queue.Empty:
                    # Continue if no frame available yet
                    continue

        except Exception as e:
            print(f"Error in display loop: {e}")
            self.frameloop = False

    def frame_grabber(self, source, frame_queue):
        """Grab frames from the video source and add to queue."""
        cap = None
        try:
            # Initialize video capture
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"Error: Unable to open video source: {source}")
                return

            # Read initial frame to get properties
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to read initial frame")
                return

            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            print(f"Video properties: {fps}")

            # Validate FPS
            if fps <= 0:
                print("Warning: Invalid FPS detected, defaulting to 30")
                fps = 30.0

            # Store FPS for writer and calculate frame timing
            self.writerfps = fps
            adjusted_fps = fps * 1.2
            #adjusted_fps = fps
            frame_timing = 1.0 / adjusted_fps

            # Set buffer size to minimize latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            lastframe = frame
            #if self.hardware == "cpu":
                #frame = cv2.resize(frame, (854, 480), interpolation=cv2.INTER_NEAREST)
            frame_queue.put(frame)  # Put initial frame in queue

            # Main frame grabbing loop
            while self.frameloop:
                ret, frame = cap.read()

                # Handle read failures
                if not ret:
                    print("Failed to read frame, stopping frame grabber")
                    self.frameloop = False
                    break

                if frame is None:
                    print("Warning: Received None frame")
                    continue

                #if self.hardware == "cpu":
                   # frame = cv2.resize(frame, (854, 480), interpolation=cv2.INTER_NEAREST)

                # Add frame to queue
                frame_queue.put(frame)

                # Control frame rate
                time.sleep(frame_timing)

            # Write sentinel frames to prevent thread lockup
            for _ in range(2):
                frame_queue.put(lastframe)

        except Exception as e:
            print(f"Error in frame_grabber: {e}")
            self.frameloop = False

        finally:
            # Ensure resources are released
            if cap is not None:
                cap.release()
            print("Frame grabber thread terminated.")

    def detection_processor(self, model, frame_queue, result_queue, write_queue, video_path=None):
        """Process frames with YOLO model and detect jumps."""
        verbose = self.tk_model_verbose.get()

        # Initialize with first frame
        first_frame = frame_queue.get()
        frame_height, frame_width, _ = first_frame.shape
        last_frame = first_frame
        frame_number = 0
        # FPS tracking
        fps_history = deque(maxlen=150)
        while self.frameloop:
            time_start = time.time()
            frame = frame_queue.get()
            frame_number += 1

            # Run YOLO detection
            if self.hardware == "cuda":
            #GPU
                #results = model(source=frame,verbose=verbose, device=self.hardware, stream_buffer=True, conf=0.35, imgsz=416, max_det=5, agnostic_nms=True, iou=0.5)
                results = model(source=frame,verbose=verbose, device=self.hardware, half=True)
            # CPU
            else:
                results = model(source=frame,verbose=verbose, device=self.hardware, stream_buffer=True, conf=0.35, imgsz=416, max_det=5, agnostic_nms=True, iou=0.5, int8=True)
            # CPU Test
            # results = model(source=frame,verbose=verbose, device=self.hardware, stream_buffer=True, conf=0.35, imgsz=416, max_det=5, agnostic_nms=True, iou=0.5, int8=True)
            detections = results[0].boxes

            # Draw counter overlay on frame
            counter_text = str(self.counter).zfill(2)
            counter_bg_x1 = int(0.5 * frame_width)
            counter_bg_y1 = int(0.064 * frame_height)
            counter_bg_x2 = int(0.628 * frame_width)
            counter_bg_y2 = int(0.098 * frame_height)
            cv2.rectangle(frame, (counter_bg_x1, counter_bg_y1),
                          (counter_bg_x2, counter_bg_y2), (0, 0, 0), -1)

            counter_text_x = int(0.5 * frame_width)
            counter_text_y = int(0.1 * frame_height)
            cv2.putText(frame, f'Counter: {counter_text}', (counter_text_x, counter_text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Display Model used
            model_text_x = int(0.02 * frame_width)
            model_text_y = int(0.98 * frame_height)
            cv2.putText(frame, f'Device : {self.hardware} // Model : {self.model_path}', (model_text_x, model_text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Process detections if any found
            if len(detections) > 0:
                best_detection = max(detections, key=lambda x: x.conf)

                # Extract bounding box coordinates
                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = map(int, best_detection.xyxy[0])
                confidence = float(best_detection.conf)
                bbox_height = bbox_y2 - bbox_y1

                # Calculate center position
                center_x = int((bbox_x1 + bbox_x2) / 2)
                center_y = int((bbox_y1 + bbox_y2) / 2)
                center_position = (center_x, center_y)
                if confidence < self.model_confidence:
                    if self.tk_save_lowscores.get():
                        self.save_lowscores(frame, frame_number, confidence, video_path)
                    continue

                # Only process high-confidence detections
                if confidence > self.model_confidence:
                    # Draw bounding box
                    cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2),
                                  (0, 255, 0), 4)

                    # Draw confidence label
                    label_text = f'Filian[{round(confidence, 1)}]'
                    label_y = bbox_y1 - 10
                    cv2.putText(frame, label_text, (bbox_x1, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

                    # Check for jumps and draw trail
                    self.jump_check(center_position, bbox_height)
                    self.trailing_dot(center_position, frame)

            # Queue frame for display and optional video writing
            result_queue.put(frame)
            if video_path is not None:
                write_queue.put(frame)

            # Calculate and display FPS
            time_end = time.time()
            time_elapsed = time_end - time_start
            fps = 1.0 / time_elapsed if time_elapsed > 0 else 0

            # Track FPS history for averaging
            fps_history.append(fps)
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
            avg_fps=round(avg_fps)

            # Draw FPS overlay on frame
            fps_text_x = int(0.02 * frame_width)
            fps_text_y = int(0.1 * frame_height)
            cv2.putText(frame, f'FPS: {avg_fps}', (fps_text_x, fps_text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Write sentinel frames to prevent thread lockup
        if video_path is not None:
            write_queue.put(last_frame)
            write_queue.put(last_frame)
        print("Detection processing thread terminated.")

    def frame_writer(self, write_queue, video_path):
        """Write processed frames to output the video file."""
        out = None
        try:
            # Get first frame to determine video dimensions
            frame = write_queue.get()
            if frame is None:
                print("Error: No frame available for video writer initialization")
                return

            frame_height, frame_width, _ = frame.shape

            # Setup output video file
            video_path_out = f'{video_path}_out.mp4'
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path_out, fourcc, int(self.writerfps), (frame_width, frame_height))

            if not out.isOpened():
                print(f"Error: Unable to open video writer for {video_path_out}")
                return

            print(f"Writing video to: {video_path_out}")

            # Write first frame
            out.write(frame)

            # Write remaining frames
            while self.frameloop:
                frame = write_queue.get()
                if frame is not None:
                    out.write(frame)

            print(f"Video saved successfully: {video_path_out}")

        except Exception as e:
            print(f"Error in frame_writer: {e}")

        finally:
            # Ensure video writer is released
            if out is not None:
                out.release()
            print("Frame writer thread terminated.")

    def cleanup_scanning(self, grabber_thread, processor_thread, writer_thread, queueref):
        """Clean up resources after scanning completes."""
        try:
            # Stop timer
            self.stop_timer()

            # Wait for threads to complete
            print("Waiting for threads to complete...")

            if processor_thread and processor_thread.is_alive():
                processor_thread.join(timeout=5)

            if grabber_thread and grabber_thread.is_alive():
                grabber_thread.join(timeout=5)

            if writer_thread and writer_thread.is_alive():
                print("Waiting for video writer to finish...")
                writer_thread.join(timeout=10)

            # Close OpenCV windows
            cv2.destroyAllWindows()

            # Reset counter trigger
            self.counter_trigger = False

            # Notify task completion
            self.task_queue.put(str(queueref))

            print("Scanning cleanup completed.")

        except Exception as e:
            print(f"Error during cleanup: {e}")

    # Processing Methods
    def jump_check(self, current_pos, bboxheight):
        """Determine if a jump has occurred based on vertical position changes."""
        bboxscale = int(bboxheight / self.relative_jump_threshold)
        self.ypos.append(current_pos[1])

        # Check for downward movement (landing)
        if self.ypos[-1] > self.ypos[0] and self.counter_trigger:
            self.counter_trigger = False
            self.delta_counter(1)

        # Check for upward movement (jumping)
        if self.ypos[-1] < (self.ypos[0] - bboxscale):
            self.counter_trigger = True

    def trailing_dot(self, current_pos, frame):
        """Draw trailing dots to visualize movement."""
        self.xypos.append(current_pos)

        for i in range(1, len(self.xypos)):
            point = self.xypos[i]
            thickness = int(10 * (i / float(len(self.xypos))))
            cv2.circle(frame, (int(point[0] + 100), int(point[1])), 1, (0, 0, 255), thickness)

    def save_lowscores(self, frame, frame_number, confidence, video_path=None):
        """Save frames with low confidence scores for training purposes."""
        # Determine output directory
        if video_path is None:
            directory_path = "Saved_Frames"
        else:
            directory_path = os.path.dirname(video_path)

        # Create directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)

        # Format filename with zero-padded frame number and formatted confidence
        confidence_str = f"{confidence:.2f}"
        filename = os.path.join(directory_path, f'lowscore_{frame_number:06d}_c-{confidence_str}.jpg')

        # Save frame
        success = cv2.imwrite(filename, frame)

        if success:
            print(f"Frame saved: {filename}")
        else:
            print(f"Error: Failed to save frame to {filename}")


    # Timer Methods
    def prompt_user_time(self):
        """Prompt user to set timer value in HH:MM:SS format."""
        usertime = promptuser("Enter the desired timer value in HH:MM:SS:")

        if not usertime or usertime == "0":
            self.reset_timer()
            return

        try:
            time_parts = usertime.split(":")
            hours = int(time_parts[0])
            minutes = int(time_parts[1])
            seconds = int(time_parts[2])
            self.set_time(hours, minutes, seconds)
        except (ValueError, IndexError):
            print("Invalid time format. Please use HH:MM:SS")

    def set_time(self, hours, minutes, seconds):
        """Set timer to specified time."""
        self.current_time = hours * 3600 + minutes * 60 + seconds
        self.update_display()

    def update_display(self):
        """Update timer display with current time."""
        hours = self.current_time // 3600
        minutes = (self.current_time % 3600) // 60
        seconds = self.current_time % 60
        self.time_label.config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")

    def start_timer(self):
        """Start the timer if not already running."""
        if not self.timer_running:
            self.timer_running = True

    def stop_timer(self):
        """Stop the timer."""
        self.timer_running = False

    def reset_timer(self):
        """Reset timer to 00:00:00."""
        self.stop_timer()
        self.set_time(0, 0, 0)

    def iterate_time(self):
        """Increment timer by one second and update display."""
        if self.timer_running:
            self.current_time += 1
            self.update_display()

        # Schedule next update in 1000ms (1 second)
        self.after(1000, self.iterate_time)

    def jumps_per_second(self):
        """Calculate and update jumps per second display."""
        if self.current_time > 0:
            jumps_per_second = self.counter / self.current_time
            self.tk_jps.set(f"{jumps_per_second:.2f}")
        else:
            self.tk_jps.set("0.00")

        self.after(1000, self.jumps_per_second)

if __name__ == "__main__":
    try:
        app = MyApp()
        app.mainloop()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user (Ctrl+C)")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        cv2.destroyAllWindows()
    finally:
        print("Application closed.")