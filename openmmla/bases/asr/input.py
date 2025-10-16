import os

from openmmla.utils.clean import flush_input
from openmmla.utils.input import interactive_menu, LIGHT_BLUE, ENDC, BLUE, GREEN, RED, YELLOW, BOLD, UNDERLINE, CYAN, GREY, WHITE, REVERSE, PINK


def get_base_mode():
    """Get the operating mode from user input."""
    options = [
        "Record Mode",
        "Recognize Mode", 
        "Full Mode"
    ]
    descriptions = [
        "Store audio locally without recognizing",
        "Recognize locally stored audio without recording",
        "Record and recognize on-the-fly"
    ]
    
    selected_index = interactive_menu("Select Operating Mode", options, descriptions)
    
    if selected_index == -1:
        raise KeyboardInterrupt("Operation cancelled")
    
    mode_map = ['record', 'recognize', 'full']
    return mode_map[selected_index]

def get_synchronizer_mode():
    """Get the operating mode from user input."""
    options = [
        "Recognize Mode",
        "Full Mode"
    ]
    descriptions = [
        "Recognize locally stored audio",
        "Recognize on-the-fly"
    ]
    
    selected_index = interactive_menu("Select Synchronizer Mode", options, descriptions)
    
    if selected_index == -1:
        raise KeyboardInterrupt("Operation cancelled")
    
    mode_map = ['recognize', 'full']
    return mode_map[selected_index]

def get_name():
    """Get the name of the speaker from user input."""
    flush_input()
    return input("Please enter the name of the speaker: ")


def get_base_type(config: dict) -> str:
    """Get the base type from user input."""
    # base types is the keys of the config dictionary and start with 'Base_'    
    base_types = [key for key in config.keys() if key.startswith('Base_')]
    
    if not base_types:
        raise ValueError("No base types found in configuration")
    
    selected_index = interactive_menu("Select Base Type", base_types)
    
    if selected_index == -1:
        raise KeyboardInterrupt("Operation cancelled")
    
    return base_types[selected_index]


def get_id():
    """Get the unique base id from user input."""
    while True:
        try:
            flush_input()
            print("------------------------------------------------")
            return int(input("Enter the your base id: "))
        except ValueError:
            print("Invalid input. Please enter an integer as your unique base id.")
            

def get_interactive_file(base_dir: str) -> tuple[str, float]:
    """Interactive file browser with cursor navigation and directory traversal.
    
    Args:
        base_dir: The base directory to start browsing from
        
    Returns:
        Tuple of (selected_file_path, initial_sync_time)
    """
    import re
    
    audio_extensions = {'.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg', '.wma'}
    current_dir = base_dir
    
    def display_file_browser(items, selected_index, current_dir):
        """Display the file browser interface."""
        clear_screen()
        print("=" * 80)
        print(f"{PINK}{BOLD}📁 Current directory: {current_dir}{ENDC}")
        print("=" * 80)
        print(f"{PINK}Use ↑/↓ arrows to navigate, Enter to select, 'q' to quit, 'b' to go back{ENDC}")
        print("-" * 80)
        
        if not items:
            print(f"{RED}No audio files or directories found in this location.{ENDC}")
            return
        
        for i, (name, item_type, path) in enumerate(items):
            if i == selected_index:
                # Highlighted selected item
                icon = "📁" if item_type == 'directory' else "🎵"
                print(f"{GREEN}{BOLD}▶ {icon} {name} ◀{ENDC}")
            else:
                # Normal item
                icon = "📁" if item_type == 'directory' else "🎵"
                if item_type == 'directory':
                    print(f"  {BLUE}{icon} {name}{ENDC}")
                else:
                    print(f"  {icon} {name}")
        
        print("-" * 80)
        print(f"{PINK}Commands: ↑/↓ = navigate, Enter = select, 'q' = quit, 'b' = back{ENDC}")
    
    while True:
        try:
            # Get all items in current directory
            items = []
            for item in sorted(os.listdir(current_dir)):
                item_path = os.path.join(current_dir, item)
                if os.path.isdir(item_path):
                    items.append((item, 'directory', item_path))
                elif any(item.lower().endswith(ext) for ext in audio_extensions):
                    items.append((item, 'file', item_path))
            
            if not items:
                clear_screen()
                print("No audio files or directories found in this location.")
                return None, None
            
            # Interactive navigation
            selected_index = 0
            while True:
                display_file_browser(items, selected_index, current_dir)
                
                key = get_key()
                
                if key == 'UP' or key == 'k':
                    selected_index = (selected_index - 1) % len(items)
                elif key == 'DOWN' or key == 'j':
                    selected_index = (selected_index + 1) % len(items)
                elif key == '\r' or key == '\n':  # Enter key
                    break
                elif key == 'q':
                    return None, None
                elif key == 'b':
                    # Go back to parent directory
                    parent_dir = os.path.dirname(current_dir)
                    if parent_dir != current_dir:  # Make sure we don't go above base_dir
                        current_dir = parent_dir
                        break
                    else:
                        clear_screen()
                        print("Already at the base directory. Press any key to continue...")
                        get_key()
                elif key == '\x03':  # Ctrl+C
                    raise KeyboardInterrupt
            
            # Handle selection
            name, item_type, path = items[selected_index]
            
            if item_type == 'directory':
                # Enter subdirectory
                current_dir = path
                continue
            else:
                # Selected a file
                selected_file = path
                
                # Extract timestamp from filename
                filename = os.path.basename(selected_file)
                match = re.search(r'_(\d+(?:\.\d+)?)\.', filename)
                default_sync_time = float(match.group(1)) if match else None
                
                # Ask for initial_sync_time
                clear_screen()
                print("=" * 80)
                print(f"{GREEN}{BOLD}Selected file: {filename}{ENDC}")
                print("=" * 80)
                
                if default_sync_time is not None:
                    print(f"Detected timestamp in filename: {default_sync_time}")
                    sync_input = input(f"Enter initial_sync_time (press Enter to use {default_sync_time}): ").strip()
                    
                    if sync_input == "":
                        initial_sync_time = default_sync_time
                    else:
                        try:
                            initial_sync_time = float(sync_input)
                            # validate that initial_sync_time is not smaller than file start time
                            if initial_sync_time < default_sync_time:
                                print(f"Error: initial_sync_time ({initial_sync_time}) cannot be smaller than file start time ({default_sync_time})")
                                print("Please try again.")
                                continue
                        except ValueError:
                            print("Invalid timestamp format. Using default.")
                            initial_sync_time = default_sync_time
                else:
                    print("No timestamp detected in filename.")
                    sync_input = input("Enter initial_sync_time: ").strip()
                    try:
                        initial_sync_time = float(sync_input)
                    except ValueError:
                        print("Invalid timestamp format. Please try again.")
                        continue
                
                return selected_file, initial_sync_time
                
        except KeyboardInterrupt:
            clear_screen()
            print("Operation cancelled.")
            return None, None
        except Exception as e:
            clear_screen()
            print(f"Error: {e}")
            return None, None

def get_input_device_index(available_indexes: list[int], device_info_list: list[dict] = None) -> int:
    """Get the input device index from user input using interactive menu.
    
    Args:
        available_indexes: List of available device indexes
        device_info_list: Optional list of device info dictionaries for display
        
    Returns:
        Selected device index
    """
    if not available_indexes:
        raise ValueError("No available input devices found")
    
    # Create options with device names and descriptions
    options = []
    descriptions = []
    
    for i, device_index in enumerate(available_indexes):
        if device_info_list and i < len(device_info_list):
            device_info = device_info_list[i]
            device_name = device_info.get('name', f'Device {device_index}')
            max_channels = device_info.get('maxInputChannels', 1)
            options.append(f"Device {device_index}: {device_name}")
            descriptions.append(f"Index: {device_index}, Max Channels: {max_channels}")
        else:
            options.append(f"Device {device_index}")
            descriptions.append(f"Index: {device_index}")
    
    selected_index = interactive_menu("Select Input Device", options, descriptions)
    
    if selected_index == -1:
        raise KeyboardInterrupt("Operation cancelled")
    
    return available_indexes[selected_index]


def get_rtmp_url(available_urls: list[str]) -> str:
    """Get the RTMP URL from user input using interactive menu.
    
    Args:
        available_urls: List of available RTMP URLs
        
    Returns:
        Selected RTMP URL
    """
    if not available_urls:
        raise ValueError("No available RTMP URLs found")
    
    # Create options with URL descriptions
    options = []
    descriptions = []
    
    for i, url in enumerate(available_urls):
        options.append(f"RTMP Stream {i + 1}")
        descriptions.append(f"URL: {url}")
    
    selected_index = interactive_menu("Select RTMP URL", options, descriptions)
    
    if selected_index == -1:
        raise KeyboardInterrupt("Operation cancelled")
    
    return available_urls[selected_index]


def get_function_base(id: int, mode: str):
    """Get the function to be performed from user input for base."""
    options = [
        "Register Speaker Profiles",
        "Start Voice Recognition", 
        f"Reset (id: {LIGHT_BLUE}{id}{ENDC})",
        f"Switch Mode (mode: {LIGHT_BLUE}{mode}{ENDC})"
    ]
    descriptions = [
        "Record and register new speaker profiles",
        "Start real-time voice recognition",
        "Reset the ASR base configuration",
        "Switch between record/recognize/full modes"
    ]
    
    selected_index = interactive_menu("Select Function", options, descriptions)

    if selected_index == -1:
        raise KeyboardInterrupt("Operation cancelled")
    
    # Map to function numbers (no exit option needed)
    function_map = [1, 2, 3, 4]
    return function_map[selected_index]


def get_function_synchronizer(mode: str):
    """Get the function to be performed from user input for synchronizer."""
    options = [
        "Start",
        f"Switch Mode (mode: {LIGHT_BLUE}{mode}{ENDC})",
        "Reset"
    ]
    descriptions = [
        "Start the synchronizer",
        "Switch between recognize/full modes",
        "Reset the ASR synchronizer configuration"
    ]
    
    selected_index = interactive_menu("Select Synchronizer Function", options, descriptions)
    
    if selected_index == -1:
        raise KeyboardInterrupt("Operation cancelled")
    
    # Map to function numbers (no exit option needed)
    function_map = [1, 2, 3]
    return function_map[selected_index]


def get_channel_selection(device_info: dict) -> int | None:
    """Get the channel selection for stereo devices using interactive menu.

    Args:
        device_info: PyAudio device info dictionary.
    Returns:
        Channel selection: channel index.
    """
    device_channels = device_info.get('maxInputChannels', 1)
    
    if device_channels <= 1:
        return None  # No channel selection needed for mono devices
    
    # Create options for each channel
    options = []
    descriptions = []
    
    for i in range(device_channels):
        options.append(f"Channel {i}")
        descriptions.append(f"Use channel {i} only")
    
    selected_index = interactive_menu(f"Select Channel (Device has {device_channels} channels)", options, descriptions)
    
    if selected_index == -1:
        raise KeyboardInterrupt("Operation cancelled")
    
    return selected_index
