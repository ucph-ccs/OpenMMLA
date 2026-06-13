from openmmla.utils.clean import flush_input
from openmmla.utils.input import interactive_menu, multi_interactive_menu, LIGHT_BLUE, ENDC, PURPLE, GREY


def get_function_base(id: int, mode: str):
    """Get the function to be performed from user input for base."""
    options = [
        "Edit Speaker Profiles",    
        "Start", 
        f"Switch Mode (mode: {LIGHT_BLUE}{mode}{ENDC})",
        f"Reset (id: {LIGHT_BLUE}{id}{ENDC})",
    ]
    descriptions = [
        "Register, select, or delete speaker profiles",
        "Start the ASR base to record and recognize audio",
        "Switch the ASR base mode between between recording, recognizing and full mode",
        "Reset the ASR base (reload config)",
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
        "Start the ASR synchronizer to synchronize the ASR bases recognition results",
        "Switch the ASR synchronizer mode between recognizing and full mode",
        "Reset the ASR synchronizer (reload config)"
    ]
    
    selected_index = interactive_menu("🎯 Select Synchronizer Function", options, descriptions, exit_on_q=True)
    
    # Map to function numbers (no exit option needed)
    function_map = [1, 2, 3]
    return function_map[selected_index]



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
    
    selected_index = interactive_menu("Select Operating Mode", options, descriptions, prompt_enter=False)
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
    
    selected_index = interactive_menu("Select Synchronizer Mode", options, descriptions, prompt_enter=False)
    mode_map = ['recognize', 'full']
    return mode_map[selected_index]


def get_name():
    """Get the name of the speaker from user input."""
    flush_input()
    return input("Please enter the name of the speaker: ")


def get_base_type(config: dict) -> str:
    """Get the base type from user input."""
    base_types = list(config.get('Base', {}).keys())

    if not base_types:
        raise ValueError("No base types found in configuration")

    selected_index = interactive_menu("Select Base Type", base_types, prompt_enter=False)
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
    
    selected_index = interactive_menu("Select Input Device", options, descriptions, prompt_enter=False)
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
    
    selected_index = interactive_menu(f"Select Channel (Device has {device_channels} channels)", options, descriptions, prompt_enter=False)
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
    
    selected_index = interactive_menu("🎯 Edit Speaker Profiles", options, descriptions, prompt_enter=False)
    return selected_index


def get_speaker_selection(available_speakers: list[str], selected_speakers: list[str]) -> list[str]:
    """Get speaker selection using interactive menu with multi-select.
    
    Args:
        available_speakers: List of all available speaker names
        selected_speakers: List of currently selected speaker names
        
    Returns:
        List of selected speaker names
    """
    selected_indices = multi_interactive_menu("🎯 Select Speakers", available_speakers, exit_on_q=False, 
                                             initial_selection=selected_speakers, include_toggle_all=False, prompt_enter=False)
    return [available_speakers[i] for i in selected_indices]


def get_speaker_deletion(available_speakers: list[str]) -> list[str]:
    """Get speakers to delete using interactive menu with multi-select.
    
    Args:
        available_speakers: List of all available speaker names
        
    Returns:
        List of speaker names to delete, or empty list if cancelled
    """
    selected_indices = multi_interactive_menu("🎯 Delete Speaker Profiles", available_speakers, exit_on_q=False, 
                                             initial_selection=[], include_toggle_all=False, prompt_enter=False)
    return [available_speakers[i] for i in selected_indices]
