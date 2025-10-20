import os
from openmmla.utils.clean import flush_input
from openmmla.utils.input import interactive_menu, LIGHT_BLUE, ENDC, PURPLE, GREEN, GREY, BOLD


def get_function_base(id: int, mode: str):
    """Get the function to be performed from user input for base."""
    options = [
        "Edit Speaker Profiles",    
        "Start Voice Recognition", 
        f"Reset (id: {LIGHT_BLUE}{id}{ENDC})",
        f"Switch Mode (mode: {LIGHT_BLUE}{mode}{ENDC})"
    ]
    descriptions = [
        "Register, select, or delete speaker profiles",
        "Start real-time voice recognition",
        "Reset the ASR base configuration",
        "Switch between record/recognize/full modes"
    ]
    
    selected_index = interactive_menu("🎯 Select Base Function", options, descriptions, exit_on_q=True)
    
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
    
    selected_index = interactive_menu("🎯 Select Synchronizer Function", options, descriptions, exit_on_q=True)
    
    # Map to function numbers (no exit option needed)
    function_map = [1, 2, 3]
    return function_map[selected_index]


def get_function_post(selected_speaker_files: list[str] = None, selected_files: list[str] = None) -> int:
    """Get the function selection for ASR Post Analyzer using interactive menu."""
    # Default to empty lists if None
    if selected_speaker_files is None:
        selected_speaker_files = []
    if selected_files is None:
        selected_files = []
    
    # Create options with current selection status in parentheses
    speaker_count = len(selected_speaker_files)
    files_count = len(selected_files)
    
    options = [
        f"Select Speaker Profiles ({LIGHT_BLUE}{speaker_count} selected{ENDC})",
        f"Select Audio Files ({LIGHT_BLUE}{files_count} selected{ENDC})", 
        "Start Processing"
    ]
    
    descriptions = [
        "Choose speaker audio files for recognition",
        "Select audio files to analyze",
        "Begin processing selected files with current settings"
    ]
    
    selected_index = interactive_menu("🎯 Select Post Analyzer Function", options, descriptions, exit_on_q=True)
    return selected_index + 1  # Return 1, 2, or 3


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
    return base_types[selected_index]


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
    return available_indexes[selected_index]


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
    return selected_index


def get_edit_speaker_options(available_speakers: list[str], selected_speakers: list[str]) -> int:
    """Get the edit speaker option using interactive menu.
    
    Args:
        available_speakers: List of all available speaker names
        selected_speakers: List of currently selected speaker names
        
    Returns:
        Selected option index (0: Register from Stream, 1: Register from Files, 2: Select/Deselect, 3: Delete)
    """
    selected_count = len(selected_speakers)
    total_count = len(available_speakers)
    
    options = [
        f"Register from Stream ({PURPLE}Record new speaker profile{ENDC})",
        f"Register from Files ({PURPLE}Use reference audio files{ENDC})",
        f"Select/Deselect Speakers ({LIGHT_BLUE}{selected_count}/{total_count} selected{ENDC})",
        f"Delete Speaker Profile ({GREY}Remove speaker permanently{ENDC})"
    ]
    
    descriptions = [
        "Record audio stream and register new speaker profile",
        "Select reference audio files to register speaker profile",
        "Choose which speakers to use for recognition",
        "Permanently delete a speaker profile from disk"
    ]
    
    selected_index = interactive_menu("🎯 Edit Speaker Profiles", options, descriptions)
    return selected_index


def _multi_select_menu(title: str, items: list[str], initial_selection: list[str], 
                      confirm_text: str = "Confirm Selection") -> list[str]:
    """Generic multi-select menu with Space key support.
    
    Args:
        title: Menu title
        items: List of items to select from
        initial_selection: List of initially selected items
        confirm_text: Text for the confirm button
        
    Returns:
        List of selected items
    """
    if not items:
        print(f"{GREY}No items available for selection.{ENDC}")
        return []
    
    # Create options with selection status
    options = []
    descriptions = []
    
    for item in items:
        if item in initial_selection:
            options.append(f"{GREEN}✓ {item}{ENDC}")
            descriptions.append("Selected - Press Space to deselect")
        else:
            options.append(f"  {item}")
            descriptions.append("Not selected - Press Space to select")
    
    # Add confirm option
    options.append(f"{GREEN}{confirm_text}{ENDC}")
    descriptions.append("Finish selecting items")
    
    selected_items_copy = initial_selection.copy()
    selected_index = 0
    
    while True:
        # Update options with current selection status
        for i, item in enumerate(items):
            if item in selected_items_copy:
                options[i] = f"{GREEN}✓ {item}{ENDC}"
                descriptions[i] = "Selected - Press Space to deselect"
            else:
                options[i] = f"  {item}"
                descriptions[i] = "Not selected - Press Space to select"
        
        # Display the menu
        from openmmla.utils.input import clear_screen, get_key
        clear_screen()
        print("=" * 80)
        print(f"{PURPLE}{BOLD}{title}{ENDC}")
        print("=" * 80)
        print(f"{PURPLE}Use ↑/↓ arrows to navigate, Space to toggle selection, Enter to confirm, 'q' to go back{ENDC}")
        print("-" * 80)
        
        for i, (option, desc) in enumerate(zip(options, descriptions)):
            if i == selected_index:
                # Highlighted selected option
                print(f"{GREEN}{BOLD}▶ {option} ◀{ENDC}")
                if desc:
                    print(f"   {GREY}└─ {desc}{ENDC}")
            else:
                # Normal option
                print(f"  {option}")
                if desc:
                    print(f"   {GREY}└─ {desc}{ENDC}")
        
        print("-" * 80)
        print(f"{PURPLE}Commands: ↑/↓ = navigate, Space = toggle, Enter = confirm, 'q' = go back{ENDC}")
        
        key = get_key()
        
        if key == 'UP' or key == 'k':
            selected_index = (selected_index - 1) % len(options)
        elif key == 'DOWN' or key == 'j':
            selected_index = (selected_index + 1) % len(options)
        elif key == ' ':  # Space - toggle selection
            if selected_index < len(items):
                # Toggle item selection
                item = items[selected_index]
                if item in selected_items_copy:
                    selected_items_copy.remove(item)
                else:
                    selected_items_copy.append(item)
            # Continue to show updated menu
        elif key == '\r' or key == '\n':  # Enter - confirm selection
            if selected_index == len(items):
                # Confirm selection
                return selected_items_copy
            # If Enter is pressed on an item, do nothing (no toggle)
            # Only Space should toggle selection
        elif key == 'q':
            raise KeyboardInterrupt("Operation Cancelled")
        elif key == '\x03':  # Ctrl+C
            raise KeyboardInterrupt


def get_speaker_selection(available_speakers: list[str], selected_speakers: list[str]) -> list[str]:
    """Get speaker selection using interactive menu with multi-select.
    
    Args:
        available_speakers: List of all available speaker names
        selected_speakers: List of currently selected speaker names
        
    Returns:
        List of selected speaker names
    """
    return _multi_select_menu("🎯 Select Speakers", available_speakers, selected_speakers, "Confirm Selection")


def get_speaker_deletion(available_speakers: list[str]) -> list[str]:
    """Get speakers to delete using interactive menu with multi-select.
    
    Args:
        available_speakers: List of all available speaker names
        
    Returns:
        List of speaker names to delete, or empty list if cancelled
    """
    return _multi_select_menu("🎯 Delete Speaker Profiles", available_speakers, [], "Confirm Deletion")
