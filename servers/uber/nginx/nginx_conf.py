import re
import socket
import sys


def resolve_ip(hostname):
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror:
        return None


def process_nginx_conf(template_path, server_names, output_path):
    # Resolve IPs
    server_ips = {name: resolve_ip(f"{name}.local") for name in server_names}

    # Read template
    with open(template_path, 'r') as f:
        content = f.read()

    # Process each server
    for name, ip in server_ips.items():
        if ip:
            # Replace placeholder with IP
            content = re.sub(f'\\${name}', ip, content)
        else:
            # Comment out the server line
            content = re.sub(f'^(\\s*server\\s*\\${name}.*)', r'#\1', content, flags=re.MULTILINE)

    # Write processed content to output file
    with open(output_path, 'w') as f:
        f.write(content)

    # Print resolved IPs
    for name, ip in server_ips.items():
        print(f"{name.capitalize()} IP: {ip or 'Not resolved'}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python process_nginx_conf.py <template_path> <output_path> <server1> [server2] ...")
        sys.exit(1)

    template_path = sys.argv[1]
    output_path = sys.argv[2]
    server_names = sys.argv[3:]

    process_nginx_conf(template_path, server_names, output_path)
