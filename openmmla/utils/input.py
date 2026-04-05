import os
import re
import sys
import tty
import termios
from datetime import datetime, timezone

from .clean import flush_input
from .client import InfluxDBClientWrapper
from .validation import validate_unix_timestamp

# Color constants for interactive UI
PINK = '\033[95m'
PURPLE = '\033[35m'
GREEN = '\033[92m'
GREY = '\033[90m'
BOLD = '\033[1m'
ENDC = '\033[0m'
BLUE = '\033[94m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
WHITE = '\033[97m'
LIGHT_BLUE = '\033[94m'
UNDERLINE = '\033[4m'
REVERSE = '\033[7m'


def get_key():
    """Get a single keypress from stdin."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        # Handle arrow keys (multi-character sequences)
        if ch == '\x1b':
            ch = sys.stdin.read(1)
            if ch == '[':
                ch = sys.stdin.read(1)
                if ch == 'A':
                    return 'UP'
                elif ch == 'B':
                    return 'DOWN'
                elif ch == 'C':
                    return 'RIGHT'
                elif ch == 'D':
                    return 'LEFT'
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name == 'posix' else 'cls')


def interactive_menu(title: str, options: list[str], descriptions: list[str] = None, exit_on_q: bool = False, prompt_enter: bool = True) -> int:
    """Interactive menu with cursor navigation and highlighting, single selection.
    
    Args:
        title: Menu title
        options: List of option names to select from
        descriptions: Optional list of descriptions for each option
        exit_on_q: Whether 'q' should exit the entire program (True) or just return to previous level (False)
        prompt_enter: Whether to prompt user to press Enter before showing the menu (default: True)
        
    Returns:
        Selected option index, or None if 'q' was pressed
    """
    if prompt_enter:
        input("\nPress Enter to continue...")
    if descriptions is None:
        descriptions = [''] * len(options)
    
    selected_index = 0
    
    while True:
        clear_screen()
        print("=" * 80)
        print(f"{PURPLE}{BOLD}{title}{ENDC}")
        print("=" * 80)
        if exit_on_q:
            print(f"{PURPLE}Use ↑/↓ arrows to navigate, Space/Enter to select, 'q' to quit{ENDC}")
        else:
            print(f"{PURPLE}Use ↑/↓ arrows to navigate, Space/Enter to select, 'q' to go back{ENDC}")
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
        if exit_on_q:
            print(f"{PURPLE}Commands: ↑/↓ = navigate, Space/Enter = select, 'q' = quit{ENDC}")
        else:
            print(f"{PURPLE}Commands: ↑/↓ = navigate, Space/Enter = select, 'q' = go back{ENDC}")
        
        key = get_key()
        
        if key == 'UP' or key == 'k':
            selected_index = (selected_index - 1) % len(options)
        elif key == 'DOWN' or key == 'j':
            selected_index = (selected_index + 1) % len(options)
        elif key == '\r' or key == '\n' or key == ' ':  # Enter or Space key to confirm
            return selected_index
        elif key == 'q':
            if exit_on_q:
                confirm = input(f"{PURPLE}Are you sure you want to exit? (y/n): {ENDC}").strip().lower()
                if confirm == 'y':
                    raise KeyboardInterrupt("Exit")
                else:
                    continue
            else:
                raise KeyboardInterrupt("Operation Cancelled")
        elif key == '\x03':  # Ctrl+C
            raise KeyboardInterrupt


def multi_interactive_menu(title: str, options: list[str], descriptions: list[str] = None, 
                           exit_on_q: bool = False, initial_selection: list[str] = None, 
                           include_toggle_all: bool = True, prompt_enter: bool = True) -> list[int]:
    """Interactive multi-select menu with toggle functionality.
    
    A general-purpose menu for selecting multiple items with:
    - Single item selection (interactive_menu): arrow keys to navigate, Enter to select, returns int (index)
    - Multiple item selection (multi_interactive_menu): arrow keys to navigate, Space to toggle, Enter to confirm, returns list[int] (indices)
    
    Args:
        title: Menu title
        options: List of option names to select from
        descriptions: Optional list of descriptions for each option
        exit_on_q: Whether 'q' should exit the entire program (True) or just return to previous level (False)
        initial_selection: List of initially selected option names (default: all selected)
        include_toggle_all: Whether to include a "Toggle All" option at the top (default: True)
        prompt_enter: Whether to prompt user to press Enter before showing the menu (default: True)
        
    Returns:
        List of selected option indices, or empty list if cancelled
    """
    if prompt_enter:
        input("\nPress Enter to continue...")
    
    if not options:
        input("No options available. Press Enter to continue...")
        return []
    
    if descriptions is None:
        descriptions = [''] * len(options)
    
    # Initialize selection state
    if initial_selection is None:
        # Default: all options selected
        selected = [True] * len(options)
    else:
        # Use provided initial selection
        selected = [option in initial_selection for option in options]
    
    current_index = 0
    
    while True:
        clear_screen()
        print("=" * 80)
        print(f"{PURPLE}{BOLD}{title}{ENDC}")
        print("=" * 80)
        if exit_on_q:
            print(f"{PURPLE}Use ↑/↓ arrows to navigate, Space to toggle, Enter to confirm, 'q' to quit{ENDC}")
        else:
            print(f"{PURPLE}Use ↑/↓ arrows to navigate, Space to toggle, Enter to confirm, 'q' to go back{ENDC}")
        print("-" * 80)
        
        # Create menu items
        menu_items = []
        menu_descriptions = []
        if include_toggle_all:
            all_selected = all(selected)
            menu_items.append(f"[{'✓' if all_selected else ' '}] Toggle All")
            menu_descriptions.append("Select or deselect all items")
        menu_items.extend([f"[{'✓' if selected[i] else ' '}] {option}" for i, option in enumerate(options)])
        menu_descriptions.extend(descriptions)
        
        for i, (item, desc) in enumerate(zip(menu_items, menu_descriptions)):
            if i == current_index:
                # Highlighted selected option
                print(f"{GREEN}{BOLD}▶ {item} ◀{ENDC}")
                if desc:
                    print(f"   {GREY}└─ {desc}{ENDC}")
            else:
                # Normal option
                print(f"  {item}")
                if desc:
                    print(f"   {GREY}└─ {desc}{ENDC}")
        
        print("-" * 80)
        selected_count = sum(selected)
        print(f"{PURPLE}Selected: {selected_count}/{len(options)} options{ENDC}")
        if exit_on_q:
            print(f"{PURPLE}Commands: ↑/↓ = navigate, Space = toggle, Enter = confirm, 'q' = quit{ENDC}")
        else:
            print(f"{PURPLE}Commands: ↑/↓ = navigate, Space = toggle, Enter = confirm, 'q' = go back{ENDC}")
        
        key = get_key()
        
        if key == 'UP' or key == 'k':
            current_index = (current_index - 1) % len(menu_items)
        elif key == 'DOWN' or key == 'j':
            current_index = (current_index + 1) % len(menu_items)
        elif key == ' ':  # Space key to toggle
            if include_toggle_all and current_index == 0:  # Toggle All option
                all_selected = all(selected)
                for i in range(len(selected)):
                    selected[i] = not all_selected
            else:  # Service options
                # Adjust index if toggle all is present
                idx = current_index - 1 if include_toggle_all else current_index
                selected[idx] = not selected[idx]
        elif key == '\r' or key == '\n':  # Enter key
            return [i for i in range(len(options)) if selected[i]]
        elif key == 'q':
            if exit_on_q:
                confirm = input(f"{PURPLE}Are you sure you want to exit? (y/n): {ENDC}").strip().lower()
                if confirm == 'y':
                    raise KeyboardInterrupt("Exit")
                else:
                    continue
            else:
                raise KeyboardInterrupt("Operation Cancelled")
        elif key == '\x03':  # Ctrl+C
            raise KeyboardInterrupt("Ctrl+C pressed")


def get_interactive_files(base_dir: str, file_extensions: tuple[str, ...] = None, multiple: bool = True,
                          sync_input: bool = False, exit_on_q: bool = False, prompt_enter: bool = True) -> list[str] | tuple[list[str], float] | tuple[str, float] | str:
    """Interactive file browser for file selection with flexible return types.
    
    Args:
        base_dir: The base directory to start browsing from
        file_extensions: Tuple of file extensions to filter (e.g., ('.wav', '.mp3'))
        multiple: Whether to allow multiple file selection
        sync_input: Whether to ask for sync time input
        exit_on_q: Whether 'q' should exit the entire program (True) or just return to previous level (False)

    Returns:
        - If multiple=True and sync_input=False: list[str]
        - If multiple=True and sync_input=True: tuple[list[str], float]
        - If multiple=False and sync_input=False: str
        - If multiple=False and sync_input=True: tuple[str, float]
        - Empty list/None if cancelled
    """
    if prompt_enter:
        input("\nPress Enter to continue...")
    current_dir = base_dir
    selected_files = []
    selected_file = None  # For single file selection mode
    
    def display_file_browser(items, selected_index, current_dir, selected_files, selected_file=None):
        """Display the file browser interface."""
        clear_screen()
        print("=" * 80)
        print(f"{PURPLE}{BOLD}📁 Current directory: {current_dir}{ENDC}")
        if multiple:
            print(f"{GREEN}Selected files: {len(selected_files)}{ENDC}")
        elif selected_file:
            print(f"{GREEN}Selected file: {os.path.basename(selected_file)}{ENDC}")
        print("=" * 80)
        
        if multiple:
            print(f"{PURPLE}Use ↑/↓ arrows to navigate, Space to select/deselect, Enter to confirm, Backspace to go back, 'q' to quit{ENDC}")
        else:
            print(f"{PURPLE}Use ↑/↓ arrows to navigate, Space to select/deselect, Enter to confirm, Backspace to go back, 'q' to quit{ENDC}")
        print("-" * 80)
        
        if not items:
            print(f"{RED}No files or directories found in this location.{ENDC}")
            return
        
        for i, (name, item_type, path) in enumerate(items):
            if i == selected_index:
                # Highlighted selected item
                icon = "📁" if item_type == 'directory' else "📄"
                print(f"{GREEN}{BOLD}▶ {icon} {name} ◀{ENDC}")
            else:
                # Normal item
                icon = "📁" if item_type == 'directory' else "📄"
                if item_type == 'directory':
                    print(f"  {BLUE}{icon} {name}{ENDC}")
                else:
                    # Check if file is selected
                    if multiple and path in selected_files:
                        print(f"  {GREEN}✓ {icon} {name}{ENDC}")
                    elif not multiple and path == selected_file:
                        print(f"  {GREEN}✓ {icon} {name}{ENDC}")
                    else:
                        print(f"  {icon} {name}")
        
        print("-" * 80)
        if multiple:
            print(f"{PURPLE}Commands: ↑/↓ = navigate, Space = select/deselect, Enter = confirm, Backspace = back, 'q' = quit{ENDC}")
        else:
            print(f"{PURPLE}Commands: ↑/↓ = navigate, Space = select/deselect, Enter = confirm, Backspace = back, 'q' = quit{ENDC}")
    
    while True:
        # Get all items in current directory
        items = []
        for item in sorted(os.listdir(current_dir)):
            item_path = os.path.join(current_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                items.append((item, 'directory', item_path))
            elif os.path.isfile(item_path) and not item.startswith('.'):
                # If file_extensions is None, show all files; otherwise filter by extensions
                if file_extensions is None or item.lower().endswith(file_extensions):
                    items.append((item, 'file', item_path))
        
        # Always add parent directory option (allow going above base_dir)
        parent_dir = os.path.dirname(current_dir)
        if parent_dir != current_dir:  # Not at filesystem root
            items.insert(0, (".. (Parent Directory)", 'directory', parent_dir))
        
        if not items:
            input("No files or directories found in this location.")
            if multiple:
                return [] if not sync_input else ([], 0.0)
            else:
                return None if not sync_input else (None, 0.0)
        
        # Interactive navigation
        selected_index = 0
        while True:
            display_file_browser(items, selected_index, current_dir, selected_files, selected_file)
            
            key = get_key()
            
            if key == 'UP' or key == 'k':
                selected_index = (selected_index - 1) % len(items)
            elif key == 'DOWN' or key == 'j':
                selected_index = (selected_index + 1) % len(items)
            elif key == ' ':  # Space - select/deselect file or enter directory
                name, item_type, path = items[selected_index]
                
                if item_type == 'directory':
                    if name == ".. (Parent Directory)":
                        current_dir = path
                        break
                    else:
                        # Enter directory
                        current_dir = path
                        break
                else:
                    if multiple:
                        # Toggle file selection
                        if path in selected_files:
                            selected_files.remove(path)
                        else:
                            selected_files.append(path)
                    else:
                        # Single file selection - toggle selection
                        if selected_file == path:
                            selected_file = None  # Deselect
                        else:
                            selected_file = path  # Select
            elif key == '\x7f':  # Backspace - go back
                # Go back to parent directory
                parent_dir = os.path.dirname(current_dir)
                if parent_dir != current_dir:  # Not at filesystem root
                    current_dir = parent_dir
                    break
                else:
                    clear_screen()
                    print("Already at filesystem root. Press any key to continue...")
                    get_key()
            elif key == '\r' or key == '\n':  # Enter - confirm selection
                if multiple:
                    if not selected_files:
                        input("No files selected. Please select at least one file.")
                        continue
                    
                    if sync_input:
                        # Ask for sync time for multiple files
                        clear_screen()
                        print("=" * 80)
                        print(f"{GREEN}{BOLD}Selected {len(selected_files)} files{ENDC}")
                        print("=" * 80)
                        
                        while True:
                            sync_input_text = input("Enter initial_sync_time: ").strip()
                            try:
                                initial_sync_time = float(sync_input_text)
                                # Validate the timestamp using validate_unix_timestamp
                                if not validate_unix_timestamp(initial_sync_time):
                                    print(f"Error: Invalid initial_sync_time ({initial_sync_time})")
                                    continue
                                return selected_files, initial_sync_time
                            except ValueError:
                                print("Invalid timestamp format. Please try again.")
                                continue
                    else:
                        return selected_files
                else:
                    # Single file selection
                    if not selected_file:
                        input("No file selected. Please select a file first.")
                        continue
                    
                    if sync_input:
                        # Ask for sync time for single file
                        filename = os.path.basename(selected_file)
                        
                        # Extract timestamp from filename
                        match = re.search(r'_(\d+(?:\.\d+)?)\.', filename)
                        default_sync_time = float(match.group(1)) if match else None
                        
                        clear_screen()
                        print("=" * 80)
                        print(f"{GREEN}{BOLD}Selected file: {filename}{ENDC}")
                        print("=" * 80)
                        
                        if default_sync_time is not None:
                            print(f"Detected timestamp in filename: {default_sync_time}")
                            
                            while True:
                                sync_input_text = input(f"Enter initial_sync_time (press Enter to use {default_sync_time}): ").strip()
                                
                                if sync_input_text == "":
                                    initial_sync_time = default_sync_time
                                else:
                                    try:
                                        initial_sync_time = float(sync_input_text)
                                        # Validate that initial_sync_time is not smaller than file start time
                                        if initial_sync_time < default_sync_time:
                                            input(f"Error: initial_sync_time ({initial_sync_time}) cannot be smaller than file start time ({default_sync_time})")
                                            continue
                                    except ValueError:
                                        input("Invalid timestamp format. Please try again.")
                                        continue
                                
                                # Validate the timestamp using validate_unix_timestamp
                                if not validate_unix_timestamp(initial_sync_time):
                                    input(f"Error: Invalid initial_sync_time ({initial_sync_time})")
                                    continue
                                    
                                return selected_file, initial_sync_time
                                    
                        else:
                            print("No timestamp detected in filename.")
                            
                            while True:
                                sync_input_text = input("Enter initial_sync_time: ").strip()
                                try:
                                    initial_sync_time = float(sync_input_text)
                                    # Validate the timestamp using validate_unix_timestamp
                                    if not validate_unix_timestamp(initial_sync_time):
                                        input(f"Error: Invalid initial_sync_time ({initial_sync_time})")
                                        continue
                                    return selected_file, initial_sync_time
                                except ValueError:
                                    input("Invalid timestamp format. Please try again.")
                                    continue
                    else:
                        return selected_file
            elif key == 'q':
                if exit_on_q:
                    # ask user to confirm exit
                    confirm = input(f"{PURPLE}Are you sure you want to exit? (y/n): {ENDC}").strip().lower()
                    if confirm == 'y':
                        raise KeyboardInterrupt("Exit")
                    else:
                        continue
                else:
                    raise KeyboardInterrupt("Operation Cancelled")
            elif key == '\x03':  # Ctrl+C
                raise KeyboardInterrupt

_SYSTEM_BUCKETS = {'_tasks', '_monitoring'}

def _is_user_bucket(name: str) -> bool:
    """return True for any bucket that is not an InfluxDB system bucket."""
    return name not in _SYSTEM_BUCKETS


def parse_bucket_timestamp(bucket_name: str) -> datetime:
    """parse the timestamp portion from a bucket name.

    Supports both new format ``<exp>_<group>_YYMMDDTHHMMZ`` and
    legacy format ``session_YYYY-MM-DDTHH:MM:SSZ``.
    """
    # new format: last segment is YYMMDDTHHMMZ
    last_seg = bucket_name.rsplit('_', 1)[-1]
    try:
        return datetime.strptime(last_seg, '%y%m%dT%H%MZ')
    except ValueError:
        pass
    # legacy: session_YYYY-MM-DDTHH:MM:SSZ
    try:
        timestamp_str = bucket_name.split('_', 1)[1]
        return datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')
    except (IndexError, ValueError):
        return datetime.min


def _format_bucket_description(name: str) -> str:
    """build a human-readable description for a bucket name."""
    # new format: <exp_id>_<group_id>_YYMMDDTHHMMZ
    parts = name.rsplit('_', 1)
    if len(parts) == 2:
        try:
            dt = datetime.strptime(parts[1], '%y%m%dT%H%MZ')
            prefix_parts = parts[0].rsplit('_', 1)
            if len(prefix_parts) == 2:
                return f"{prefix_parts[0]} / {prefix_parts[1]} — {dt.strftime('%Y-%m-%d %H:%M')} UTC"
        except ValueError:
            pass
    # legacy: session_YYYY-MM-DDTHH:MM:SSZ
    try:
        timestamp_str = name.split('_', 1)[1]
        dt = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')
        return f"Session from {dt.strftime('%Y-%m-%d %H:%M:%S')}"
    except (IndexError, ValueError):
        return "Bucket"

def select_bucket(influx_client: InfluxDBClientWrapper) -> str:
    """Get the bucket name from the user using interactive menu."""
    while True:
        bucket_list = influx_client.get_buckets()
        bucket_names = [b.name for b in bucket_list.buckets if _is_user_bucket(b.name)]

        try:
            bucket_names = sorted(bucket_names, key=parse_bucket_timestamp)
        except Exception as e:
            print(f'{YELLOW}Warning: Some buckets have incompatible formats: {e}{ENDC}')

        if not bucket_names:
            input("No bucket sessions found.")
            return None

        options = [f"📁 {name}" for name in bucket_names]
        descriptions = [_format_bucket_description(name) for name in bucket_names]

        selected_index = interactive_menu("Select Session Bucket", options, descriptions, prompt_enter=False)

        bucket_name = bucket_names[selected_index]
        clear_screen()
        print(f"{GREEN}✅ Bucket: {bucket_name} has been selected.{ENDC}")
        return bucket_name


def _make_bucket_name(exp_id: str, group_id: str, dt: datetime) -> str:
    """build a bucket name in the ``<exp_id>_<group_id>_YYMMDDTHHMMZ`` format."""
    return f"{exp_id}_{group_id}_{dt.strftime('%y%m%dT%H%MZ')}"


def select_or_create_bucket(influx_client: InfluxDBClientWrapper) -> str:
    """Get the bucket name from user input using interactive menu, either select an existing bucket or create a new one."""
    from .experiments import select_experiment_and_group

    while True:
        bucket_list = influx_client.get_buckets()
        bucket_names = [b.name for b in bucket_list.buckets if _is_user_bucket(b.name)]

        try:
            bucket_names = sorted(bucket_names, key=parse_bucket_timestamp)
        except Exception as e:
            print(f'{YELLOW}Warning: Some buckets have incompatible formats: {e}{ENDC}')

        options = [f"📁 {name}" for name in bucket_names]
        descriptions = [_format_bucket_description(name) for name in bucket_names]

        options.extend([
            "➕ Create New Bucket (Auto)",
            "⌨️  Create New Bucket (Manual)",
            "🔄 Refresh List"
        ])
        descriptions.extend([
            "Select experiment & group, use current timestamp",
            "Select experiment & group, provide custom Unix timestamp",
            "Refresh the bucket list"
        ])

        if not bucket_names:
            options = options[-3:]
            descriptions = descriptions[-3:]

        selected_index = interactive_menu("Select Session Bucket", options, descriptions)

        if selected_index < len(bucket_names):
            bucket_name = bucket_names[selected_index]
            clear_screen()
            print(f"{GREEN}✅ Bucket: {bucket_name} has been selected.{ENDC}")
            return bucket_name
        elif options[selected_index] == "➕ Create New Bucket (Auto)":
            clear_screen()
            exp_id, group_id = select_experiment_and_group()
            bucket_name = _make_bucket_name(exp_id, group_id, datetime.now(timezone.utc))
            influx_client.create_bucket(bucket_name)
            clear_screen()
            print(f"{GREEN}✅ Bucket: {bucket_name} has been created.{ENDC}")
            return bucket_name
        elif options[selected_index] == "⌨️  Create New Bucket (Manual)":
            clear_screen()
            exp_id, group_id = select_experiment_and_group()
            clear_screen()
            print("=" * 80)
            print(f"{PURPLE}{BOLD}Create New Bucket with Custom Timestamp{ENDC}")
            print("=" * 80)
            print(f"{PURPLE}Enter a Unix timestamp (e.g., 1759904435.95792 or 1759904435){ENDC}")
            print("-" * 80)

            while True:
                try:
                    user_input = input("Unix timestamp: ").strip()

                    if not user_input:
                        print(f"{RED}Timestamp cannot be empty. Please try again.{ENDC}")
                        continue

                    unix_timestamp = float(user_input)

                    if not validate_unix_timestamp(unix_timestamp):
                        print(f"{RED}Error: Invalid Unix timestamp ({unix_timestamp}){ENDC}")
                        print(f"{YELLOW}Timestamp should be a reasonable Unix time value.{ENDC}")
                        continue

                    dt = datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)
                    bucket_name = _make_bucket_name(exp_id, group_id, dt)

                    print("-" * 80)
                    print(f"{CYAN}Preview:{ENDC}")
                    print(f"  Unix timestamp: {unix_timestamp}")
                    print(f"  Human readable: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                    print(f"  Bucket name: {bucket_name}")
                    print("-" * 80)

                    confirm = input(f"{PURPLE}Create this bucket? (y/n): {ENDC}").strip().lower()
                    if confirm == 'y':
                        influx_client.create_bucket(bucket_name)
                        clear_screen()
                        print(f"{GREEN}✅ Bucket: {bucket_name} has been created.{ENDC}")
                        return bucket_name
                    else:
                        print(f"{YELLOW}Bucket creation cancelled.{ENDC}")
                        input("Press Enter to return to menu...")
                        break

                except ValueError:
                    print(f"{RED}Invalid timestamp format. Please enter a valid Unix timestamp.{ENDC}")
                    continue
                except Exception as e:
                    print(f"{RED}Error: {e}{ENDC}")
                    input("Press Enter to return to menu...")
                    break
        elif options[selected_index] == "🔄 Refresh List":
            continue


def get_id():
    """Get the unique base id from user input."""
    while True:
        try:
            flush_input()
            print("------------------------------------------------")
            return int(input("Enter the your base id: "))
        except ValueError:
            print("Invalid input. Please enter an integer as your unique base id.")


def get_number_of_bases(default: int = 1) -> int:
    """Get the number of bases to synchronize.

    Args:
        default: Default number of bases if user presses Enter

    Returns:
        int: Number of bases
    """
    while True:
        try:
            flush_input()
            prompt = f"Enter the number of bases to synchronize [default: {default}]: "
            user_input = input(prompt)
            
            # empty input uses default value
            if not user_input.strip():
                print(f"Using default value: {default}")
                return default
            
            num = int(user_input)
            if num > 0:
                return num
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")

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
    
    selected_index = interactive_menu("Select RTMP URL", options, descriptions, prompt_enter=False)    
    return available_urls[selected_index]

