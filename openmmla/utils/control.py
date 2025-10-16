"""Control utility for managing ASR, IPS, and VFA services."""
import os
import sys
import tty
import termios

from .input import interactive_menu
from .client import InfluxDBClientWrapper, RedisClientWrapper
from .logger import get_logger


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


def get_operation() -> int:
    """Get the operation to perform using interactive menu."""
    options = [
        "Reconnect nodes",
        "Disconnect nodes"
    ]
    descriptions = [
        "Send START signal to selected services",
        "Send STOP signal to selected services"
    ]
    
    selected_index = interactive_menu("Select Operation", options, descriptions)
    
    if selected_index == -1:
        raise KeyboardInterrupt("Operation cancelled")
    
    return selected_index + 1  # Return 1 or 2


def select_services() -> list[str]:
    """Display an interactive menu with cursor to choose which services to control."""
    all_services = ["asr", "ips", "vfa"]
    
    try:
        import curses
        
        def curses_menu(stdscr):
            # Initialize colors
            curses.curs_set(0)  # Hide cursor
            curses.start_color()
            curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)
            
            # Initialize selection state
            selected = [True] * len(all_services)
            current_row = 0
            
            # Menu items including Toggle All at the top
            def get_menu_items():
                items = ["[{}] Toggle All".format("x" if all(selected) else " ")]
                for i, service in enumerate(all_services):
                    items.append("[{}] {}".format("x" if selected[i] else " ", service))
                return items
            
            # Display the menu
            while True:
                stdscr.clear()
                stdscr.addstr(0, 0, "Use ↑/↓ to navigate, SPACE to toggle selection, ENTER to confirm:",
                               curses.color_pair(1))
                
                menu_items = get_menu_items()
                
                # Display menu items
                for idx, item in enumerate(menu_items):
                    if idx == current_row:
                        stdscr.addstr(idx + 2, 0, "> " + item, curses.color_pair(2))
                    else:
                        stdscr.addstr(idx + 2, 0, "  " + item, curses.color_pair(1))
                
                # Add instructions at the bottom
                stdscr.addstr(len(menu_items) + 3, 0,
                               "Note: If no services are selected, you will return to the main menu.",
                               curses.color_pair(1))
                
                stdscr.refresh()
                
                # Handle keyboard input
                key = stdscr.getch()
                
                if key == curses.KEY_UP:
                    current_row = (current_row - 1) % len(menu_items)
                elif key == curses.KEY_DOWN:
                    current_row = (current_row + 1) % len(menu_items)
                elif key == ord(' '):  # Space key to toggle
                    if current_row == 0:  # Toggle All option
                        all_selected = all(selected)
                        for i in range(len(selected)):
                            selected[i] = not all_selected
                    else:  # Service options
                        selected[current_row - 1] = not selected[current_row - 1]
                elif key == 10:  # Enter key - confirm and exit
                    break  # Always exit, even if nothing is selected
                elif key == 27:  # ESC key - cancel and exit
                    return []  # Return empty list to indicate cancellation
            
            # Return the selected services
            return [all_services[i] for i in range(len(all_services)) if selected[i]]
        
        # Run the curses menu
        selected_services = curses.wrapper(curses_menu)
        
        if not selected_services:
            print("No services selected, returning to main menu.")
            return []
            
        print(f"Selected services: {', '.join(selected_services)}")
        return selected_services
        
    except Exception as e:
        print(f"Error in service selection: {e}")
        print("Please make sure your terminal supports curses.")
        return []


def start_control(config_path: str):
    """Start control interface with restart capability.
    
    Args:
        config_path: Path to the configuration file
    """
    logger = get_logger('control')

    print(f"\033]0; Control Base \007")
    
    # Restart loop - allows restarting the entire process
    while True:
        try:
            influx_client = InfluxDBClientWrapper(config_path)
            redis_client = RedisClientWrapper(config_path)

            operation = get_operation()
            
            from .input import select_or_create_bucket
            bucket_name = select_or_create_bucket(influx_client)
            if not bucket_name:
                continue
                
            command = 'START' if operation == 1 else 'STOP'
            selected_services = select_services()
            
            if not selected_services:
                continue
            
            # Send control signal to each selected service
            for service in selected_services:
                channel = f"{bucket_name}/{service}/control"
                redis_client.publish(channel, command)
                print(f"{'✅' if command == 'START' else '🛑'} {command} signal sent to {service}.")
            
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt as e:
            if "Operation cancelled" in str(e):
                print("\n👋 Goodbye!")
                break  # Exit completely when 'q' is pressed
            else:
                print("\n🔄 Restarting Control Base...")
                continue  # Restart on Ctrl+C during runtime
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("🔄 Restarting Control Base...")
            continue  # Restart on error
