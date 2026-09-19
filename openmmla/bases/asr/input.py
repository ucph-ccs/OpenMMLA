from openmmla.utils.clean import flush_input
from openmmla.utils.config import get_bases
from openmmla.utils.input import interactive_menu, multi_interactive_menu, pause_after_error, LIGHT_BLUE, ENDC, \
    PURPLE, GREY, RED


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
        "Capture Mode",
        "Analyze Mode",
        "Live Mode"
    ]
    descriptions = [
        "Store audio locally without recognizing",
        "Recognize locally stored audio without recording",
        "Record and recognize on-the-fly"
    ]

    selected_index = interactive_menu("Select Operating Mode", options, descriptions, prompt_enter=False)
    mode_map = ['capture', 'analyze', 'live']
    return mode_map[selected_index]


def get_synchronizer_mode():
    """Get the operating mode from user input."""
    options = [
        "Analyze Mode",
        "Live Mode"
    ]
    descriptions = [
        "Recognize locally stored audio",
        "Recognize on-the-fly"
    ]

    selected_index = interactive_menu("Select Synchronizer Mode", options, descriptions, prompt_enter=False)
    mode_map = ['analyze', 'live']
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


def get_base_types(config: dict) -> list[str]:
    """The keys of the config's Base section: one block per kind of microphone."""
    base_section = (config or {}).get('Base') or {}
    return [str(key) for key in base_section] if isinstance(base_section, dict) else []


def default_base_type(config: dict) -> tuple[str | None, str]:
    """The base type an ASR synchronizer launched from the console takes when
    -bt/--base_type is not given: the only key of the Base section.

    Returns (base type, "") or, when there is no single one, (None, why).
    """
    base_types = get_base_types(config)
    if len(base_types) == 1:
        return base_types[0], ""
    if not base_types:
        return None, "the config's Base section has no entry (one block per kind of microphone)."
    return None, (f"the config's Base section has {len(base_types)} entries ({', '.join(base_types)}) "
                  "and no base type was given with -bt.")


def default_number_of_bases(config: dict) -> tuple[int | None, str]:
    """The number of bases an ASR synchronizer launched from the console waits
    for when -nb/--num_bases is not given: the entries of the config's Bases list.

    Returns (number, "") or, when the list is empty, (None, why).
    """
    count = len(get_bases(config))
    if count:
        return count, ""
    return None, "the config's Bases list is empty, so there is no number of bases to wait for."


def explain_cannot_start(component: str, why: str, fix: str, wait: bool = False):
    """Say why a process launched from the console cannot start on its own, and
    what to do in the menu or prompt that follows in the same window.

    wait: the menu that follows clears the screen as it opens (one opened with
    prompt_enter=False), which would wipe this before anyone reads it, so wait
    for Enter first. A plain prompt, or a menu that asks for Enter itself,
    leaves it on the screen and needs no wait."""
    print("------------------------------------------------")
    print(f"{RED}{component} cannot start on its own: {why}{ENDC}")
    print(fix)
    if wait:
        pause_after_error("open the menu")


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
