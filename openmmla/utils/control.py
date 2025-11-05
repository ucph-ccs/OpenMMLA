"""Control utility for managing ASR, IPS, and VFA services."""

from .input import interactive_menu, multi_interactive_menu
from .client import InfluxDBClientWrapper, RedisClientWrapper
from .logger import get_logger


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
    
    selected_index = interactive_menu("Select Operation", options, descriptions, exit_on_q=True)
    return selected_index + 1  # Return 1 or 2


def select_services() -> list[str]:
    """Display an interactive menu to choose which services to control."""
    all_services = ["asr", "ips", "vfa"]
    
    try:
        selected_indices = multi_interactive_menu("Select Services", all_services, prompt_enter=False)
        
        if not selected_indices:
            print("No services selected, returning to main menu.")
            return []
        
        selected_services = [all_services[i] for i in selected_indices]
        print(f"Selected services: {', '.join(selected_services)}")
        return selected_services
        
    except Exception as e:
        print(f"Error in service selection: {e}")
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
                
            command = 'START' if operation == 1 else 'STOP'
            selected_services = select_services()
            
            if not selected_services:
                continue
            
            # Send control signal to each selected service
            for service in selected_services:
                channel = f"{bucket_name}/{service}/control"
                redis_client.publish(channel, command)
                print(f"{'✅' if command == 'START' else '🛑'} {command} signal sent to {service}.")

        except KeyboardInterrupt as e:
            if "Exit" in str(e):
                print("\n👋 Goodbye!")
                break  # Exit completely when 'q' is pressed
            elif "Operation Cancelled" in str(e):
                print("\n🔄 Restarting Control Base...")
                continue  # Restart on 'q' pressed in lower-level menu
            else:
                print("\n🔄 Restarting Control Base...")
                continue  # Restart on Ctrl+C during runtime
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("🔄 Restarting Control Base...")
            continue  # Restart on error
