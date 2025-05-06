import argparse
import functools
import os


def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla ses-ctl",
        description="Control bucket session.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150)
    )
    from openmmla.utils.args import add_arguments
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('config_path', str, None, 'path to the configuration file', shortname='-c', required=True)
    return parser


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
        import sys
        sys.exit(1)


def run_session_control(args):
    print(f"\033]0; Control Base \007")

    from openmmla.utils.logger import get_logger
    from openmmla.utils.clean import flush_input
    from openmmla.utils.input import select_bucket
    from openmmla.utils.client import RedisClientWrapper
    from openmmla.utils.client import InfluxDBClientWrapper

    logger = get_logger('control')

    config_path = args.config_path
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.getcwd(), config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    while True:
        try:
            redis_client = RedisClientWrapper(config_path)
            influx_client = InfluxDBClientWrapper(config_path)

            flush_input()
            operation = input(
                "Please input your operation:\n"
                "1: Reconnect nodes\n"
                "2: Disconnect nodes\n"
                "0: Exit\n"
                "Selected function: "
            ).strip()

            if operation in ['1', '2']:
                bucket_name = select_bucket(influx_client)
                if not bucket_name:
                    continue

                command = 'START' if operation == '1' else 'STOP'
                selected_services = select_services()

                if not selected_services:
                    continue

                # Send control signal to each selected service
                for service in selected_services:
                    channel = f"{bucket_name}/{service}/control"
                    redis_client.publish(channel, command)
                    print(f"{'✅' if command == 'START' else '🛑'} {command} signal sent to {service}.")

            elif operation == '0':
                break
            else:
                print("Invalid operation. Please input 1, 2, or 0.")
        except (Exception, KeyboardInterrupt) as e:
            logger.warning(
                f"Interrupted: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}. Returning to main menu.",
                exc_info=True
            )


def main():
    parser = get_parser()
    args = parser.parse_args()

    from openmmla.utils.args import print_arguments
    print_arguments(args)

    run_session_control(args)


if __name__ == "__main__":
    main()
