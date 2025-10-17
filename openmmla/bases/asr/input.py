from openmmla.utils.clean import flush_input
from openmmla.utils.input import interactive_menu, LIGHT_BLUE, ENDC


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
    
    selected_index = interactive_menu("🎯 Select Base Function", options, descriptions)

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
    
    selected_index = interactive_menu("🎯 Select Synchronizer Function", options, descriptions)
    
    if selected_index == -1:
        raise KeyboardInterrupt("Operation cancelled")
    
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
    
    selected_index = interactive_menu("🎯 Select Post Analyzer Function", options, descriptions)
    
    if selected_index == -1:
        raise KeyboardInterrupt("Operation cancelled")
    
    return selected_index + 1  # Return 1, 2, or 3


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
