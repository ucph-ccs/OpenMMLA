import os
import sys
import socket
import yaml
import ipaddress
import asyncio
from jinja2 import Environment, FileSystemLoader
from concurrent.futures import ThreadPoolExecutor

import platform


# ========== 工具函数 ==========

def is_valid_ip(name):
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return False


def resolve_hostname(hostname):
    try:
        return socket.gethostbyname(f"{hostname}")
    except socket.gaierror:
        return None


def is_host_alive(ip_or_hostname, timeout=5):
    try:
        for port in [22]:  # check SSH port
            with socket.create_connection((ip_or_hostname, port), timeout=timeout):
                return True
    except Exception:
        return False


def is_ip_port_open(ip, port, timeout=5):
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except Exception:
        return False


def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


# ========== 异步解析 + 检测 ==========

async def resolve_and_check_servers(config, check_port=False, max_workers=20):
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=max_workers)

    tasks = []

    # Ensure upstreams exists in config
    if "upstreams" not in config:
        config["upstreams"] = {}
        return

    for service, servers in config.get("upstreams", {}).items():
        for server in servers:
            task = loop.run_in_executor(executor, resolve_and_check_one, server, check_port)
            tasks.append(task)

    await asyncio.gather(*tasks)


def resolve_and_check_one(server, check_port):
    host = server['host']
    port = server.get('port', 80)

    # Step 1: resolve
    if is_valid_ip(host):
        ip = host
    else:
        ip = resolve_hostname(host)

    if not ip:
        server['ip'] = None
        server['reachable'] = False
        return

    # Step 2: check port if needed
    if check_port:
        reachable = is_ip_port_open(ip, port)
    else:
        reachable = is_host_alive(ip)

    server['ip'] = ip if reachable else None
    server['reachable'] = reachable


# ========== 渲染 Jinja 模板 ==========

def render_nginx_template(config, template_path, output_path):
    env = Environment(loader=FileSystemLoader(os.path.dirname(template_path)))
    template = env.get_template(os.path.basename(template_path))
    
    # Ensure required keys exist in config
    if "upstreams" not in config:
        config["upstreams"] = {}
    
    if "rtmp_apps" not in config:
        config["rtmp_apps"] = []
    
    rendered = template.render(config=config)

    with open(output_path, 'w') as f:
        f.write(rendered)

    print("\n Server Resolution Summary:")
    print("=" * 40)
    
    if not config.get("upstreams"):
        print("ℹ️ No upstream servers defined in config.")
    else:
        for service, servers in config.get("upstreams", {}).items():
            reachable_servers = [s for s in servers if s.get("reachable")]
            if not reachable_servers:
                print(f"⚠️  Skipped upstream '{service}': no reachable servers.")
            else:
                for s in servers:
                    status = f"✅ {s['ip']}" if s.get("reachable") else "❌ Not reachable"
                    print(f"{s['host']} ({service}) → {status}")
    
    if not config.get("rtmp_apps"):
        print("ℹ️ No RTMP applications defined in config.")
    else:
        print("\n RTMP Applications:")
        print("=" * 40)
        for app in config.get("rtmp_apps", []):
            print(f"✅ {app}")


# ========== 主入口 ==========

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python render_nginx.py <config.yml> <template.j2> <output.conf> [--port-check]")
        sys.exit(1)

    config_path, template_path, output_path = sys.argv[1:4]
    check_port = "--port-check" in sys.argv

    config = load_config(config_path)
    config["is_linux"] = platform.system().lower() == "linux"

    asyncio.run(resolve_and_check_servers(config, check_port=check_port))
    render_nginx_template(config, template_path, output_path)
