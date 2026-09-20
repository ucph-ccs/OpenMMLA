import os
import sys
import socket
import yaml
import ipaddress
import asyncio
from jinja2 import Environment, FileSystemLoader
from concurrent.futures import ThreadPoolExecutor


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


def probe_port(ip, port, timeout=3):
    """how a connect to `port` on the machine at `ip` goes: "open" (a server
    listens), "refused" (the machine answers, nothing listens there yet) or
    "silent" (no answer at all: the machine is off or out of reach, or a
    firewall drops it)."""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return "open"
    except ConnectionRefusedError:
        return "refused"
    except OSError:
        return "silent"


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
        server['ip'], server['state'], server['routed'] = None, "unresolved", False
        return

    # Step 2: every server whose name resolves is routed, whatever its port
    # answers right now. nginx takes up by itself a server that starts later (one
    # that fails is skipped for fail_timeout and then tried again with a real
    # request), so what the render finds running must not decide what the config
    # holds: that is what made the Gateway need a new start after the ASR or VFA
    # server. The probe only tells the summary below what is up. A machine that
    # does not answer costs one request per fail_timeout up to
    # proxy_connect_timeout, which nginx then sends on to the next server
    server['ip'] = ip
    server['state'] = probe_port(ip, port) if check_port else "unchecked"
    server['routed'] = True


# ========== 渲染 Jinja 模板 ==========

# the summary line of a server, by how its port answered (resolve_and_check_one)
_STATES = {
    "open": "✅ {ip}",
    "refused": "⏳ {ip}: nothing on port {port} yet, nginx sends to it once it answers",
    "silent": "💤 {ip} does not answer (off, out of reach, or a firewall drops it): routed all the "
              "same, nginx skips it while it fails",
    "unresolved": "❌ the name does not resolve: left out, nginx would not start with it",
    "unchecked": "➖ {ip} (not checked)",
}


def render_nginx_template(config, template_path, output_path):
    env = Environment(loader=FileSystemLoader(os.path.dirname(template_path)))
    template = env.get_template(os.path.basename(template_path))
    
    # Ensure required keys exist in config
    if "upstreams" not in config:
        config["upstreams"] = {}

    rendered = template.render(config=config)

    with open(output_path, 'w') as f:
        f.write(rendered)

    print("\n Server Resolution Summary:")
    print("=" * 40)
    
    if not config.get("upstreams"):
        print("ℹ️ No upstream servers defined in config.")
    else:
        for service, servers in config.get("upstreams", {}).items():
            if not any(s.get("routed") for s in servers):
                print(f"⚠️  Upstream '{service}': not one of its names resolves, so /{service} answers "
                      f"503 (which the bases retry) until one does.")
            for s in servers:
                status = _STATES[s["state"]].format(ip=s["ip"], port=s.get("port", 80))
                print(f"{s['host']} ({service}) → {status}")

    if config.get("rtmp_apps"):
        print("ℹ️ rtmp_apps is ignored: streams go through MediaMTX now (docs/rtmp_streaming.md).")


# ========== 主入口 ==========

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python render_nginx.py <config.yml> <template.j2> <output.conf> [--port-check]")
        sys.exit(1)

    config_path, template_path, output_path = sys.argv[1:4]
    check_port = "--port-check" in sys.argv

    config = load_config(config_path)

    asyncio.run(resolve_and_check_servers(config, check_port=check_port))
    render_nginx_template(config, template_path, output_path)
