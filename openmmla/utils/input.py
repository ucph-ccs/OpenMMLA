import os
import sys
import tty
import termios
from datetime import datetime, timezone

from .clean import flush_input
from .client import InfluxDBClientWrapper

# Color constants for interactive UI
PINK = '\033[95m'
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


def interactive_menu(title: str, options: list[str], descriptions: list[str] = None) -> int:
    """Interactive menu with cursor navigation and highlighting.
    
    Args:
        title: Menu title
        options: List of option strings
        descriptions: Optional list of descriptions for each option
        
    Returns:
        Selected option index
    """
    if descriptions is None:
        descriptions = [''] * len(options)
    
    selected_index = 0
    
    while True:
        clear_screen()
        print("=" * 80)
        print(f"{PINK}{BOLD}🎯 {title}{ENDC}")
        print("=" * 80)
        print(f"{PINK}Use ↑/↓ arrows to navigate, Enter to select, 'q' to quit{ENDC}")
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
        print(f"{PINK}Commands: ↑/↓ = navigate, Enter = select, 'q' = quit{ENDC}")
        
        key = get_key()
        
        if key == 'UP' or key == 'k':
            selected_index = (selected_index - 1) % len(options)
        elif key == 'DOWN' or key == 'j':
            selected_index = (selected_index + 1) % len(options)
        elif key == '\r' or key == '\n':  # Enter key
            return selected_index
        elif key == 'q':
            return -1  # Quit
        elif key == '\x03':  # Ctrl+C
            raise KeyboardInterrupt


def select_bucket(influx_client: InfluxDBClientWrapper) -> str:
    """Get the bucket name from the user."""
    bucket_name = None

    while True:
        flush_input()
        print("------------------------------------------------")
        bucket_list = influx_client.get_buckets()
        bucket_names = [bucket.name for bucket in bucket_list.buckets if bucket.name not in ['_tasks', '_monitoring']]
        bucket_names = [name for name in bucket_names if 'session_' in name]

        try:
            bucket_names = sorted(bucket_names, key=lambda x: datetime.strptime(x.split('_')[1], '%Y-%m-%dT%H:%M:%SZ'))
            if len(bucket_names) == 0:
                print("No bucket sessions found.")
                break
            for i, name in enumerate(bucket_names, start=1):
                print(f"{i}. {name}")
            bucket_idx = input("Enter the number of the bucket you want to select:")
        except Exception as e:
            print(f'No compatible bucket session to sort, {e}')
            break

        if bucket_idx.isdigit() and 1 <= int(bucket_idx) <= len(bucket_names):
            bucket_name = bucket_names[int(bucket_idx) - 1]
            print(f"Bucket: {bucket_name} has been selected.")
            break
        else:
            print("Invalid input. Please enter a number from the list.")
            continue

    return bucket_name


def select_or_create_bucket(influx_client):
    """Get the bucket name from user input using interactive menu, either select an existing bucket or create a new one."""
    while True:
        try:
            # Get bucket list
            bucket_list = influx_client.get_buckets()
            bucket_names = [bucket.name for bucket in bucket_list.buckets if bucket.name not in ['_tasks', '_monitoring']]
            bucket_names = [name for name in bucket_names if 'session_' in name]

            # Sort buckets by timestamp
            try:
                bucket_names = sorted(bucket_names, key=lambda x: datetime.strptime(x.split('_')[1], '%Y-%m-%dT%H:%M:%SZ'))
            except Exception as e:
                print(f'No compatible bucket session to sort, {e}')
                bucket_names = []

            # Create options for interactive menu
            options = []
            descriptions = []
            
            # Add existing buckets
            for name in bucket_names:
                # Extract timestamp and format it nicely
                try:
                    timestamp_str = name.split('_')[1]
                    dt = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')
                    formatted_date = dt.strftime('%Y-%m-%d %H:%M:%S')
                    options.append(f"📁 {name}")
                    descriptions.append(f"Session from {formatted_date}")
                except:
                    options.append(f"📁 {name}")
                    descriptions.append("Session bucket")
            
            # Add special options
            options.extend([
                "➕ Create New Bucket",
                "🔄 Refresh List"
            ])
            descriptions.extend([
                "Create a new session bucket with current timestamp",
                "Refresh the bucket list"
            ])

            if not options:
                # No buckets available, only show create option
                options = ["➕ Create New Bucket"]
                descriptions = ["Create a new session bucket with current timestamp"]

            # Show interactive menu
            selected_index = interactive_menu("Select Session Bucket", options, descriptions)
            
            if selected_index == -1:
                raise KeyboardInterrupt("Operation cancelled")
            
            # Handle selection
            if selected_index < len(bucket_names):
                # Selected existing bucket
                bucket_name = bucket_names[selected_index]
                clear_screen()
                print(f"{GREEN}✅ Bucket: {bucket_name} has been selected.{ENDC}")
                return bucket_name
            elif options[selected_index] == "➕ Create New Bucket":
                # Create new bucket
                timestamp = datetime.now(timezone.utc).isoformat().split('.')[0] + 'Z'
                bucket_name = 'session_' + timestamp
                influx_client.create_bucket(bucket_name)
                clear_screen()
                print(f"{GREEN}✅ Bucket: {bucket_name} has been created.{ENDC}")
                return bucket_name
            elif options[selected_index] == "🔄 Refresh List":
                # Refresh list and continue loop
                continue
                
        except KeyboardInterrupt:
            clear_screen()
            print(f"{GREY}Operation cancelled.{ENDC}")
            raise
        except Exception as e:
            clear_screen()
            print(f"Error: {e}")
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


def get_number_of_bases() -> int:
    """Get the number of bases to synchronize.

    Returns:
        int: Number of bases
    """
    while True:
        try:
            flush_input()
            num = int(input("Enter the number of bases to synchronize: ") or "1")
            if num > 0:
                return num
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")


def select_participant_descriptions(participant_descriptions_config: dict) -> dict | None:
    """Select participant descriptions for the current session.
    
    Args:
        participant_descriptions_config: Dictionary of participant descriptions from config
                                       Format: {session_key: {tag_id: description}}
        
    Returns:
        dict: Selected participant descriptions as {tag_id: description} mapping, 
              or None if no selection made
    """
    if not participant_descriptions_config:
        print("No participant descriptions available in configuration")
        return None
        
    print("------------------------------------------------")
    print("Available participant description sets:")
    
    description_keys = list(participant_descriptions_config.keys())
    for idx, key in enumerate(description_keys):
        participant_count = len(participant_descriptions_config[key])
        print(f"{idx + 1}: {key} ({participant_count} participants)")
    
    print(f"{len(description_keys) + 1}: None - No participant descriptions")
    
    while True:
        try:
            flush_input()
            selection_input = input(f"Choose participant description set (1-{len(description_keys) + 1}) or press Enter for None: ")
            
            if selection_input == '':
                return None
                
            selection = int(selection_input)
            if 1 <= selection <= len(description_keys):
                selected_key = description_keys[selection - 1]
                selected_descriptions = participant_descriptions_config[selected_key]
                print(f"Selected participant descriptions: {selected_key}")
                print("Participants:")
                for tag_id, description in selected_descriptions.items():
                    print(f"  Tag ID {tag_id}: {description}")
                return selected_descriptions
            elif selection == len(description_keys) + 1:
                return None
            else:
                print("Invalid selection. Please choose a valid option.")
        except ValueError:
            print("Please enter a valid number or press Enter for None.")
