# Nginx Documentation

This document outlines the setup and configuration of Nginx for load balancing and RTMP streaming.

## Installation

Choose the appropriate installation based on your needs:

### Basic Installation (Load Balancer Only)

```bash
# macOS
brew install nginx
# config file located at `/opt/homebrew/etc/nginx/nginx.conf`

# Ubuntu
sudo apt update && install nginx
# config file located at `/etc/nginx/nginx.conf`
```

### Complete Installation (Load Balancer + RTMP)

If you need RTMP functionality, use this installation instead:

```bash
# macOS
brew tap denji/nginx
brew install nginx-full --with-rtmp-module
# config file located at `/opt/homebrew/etc/nginx/nginx.conf`

# Ubuntu
sudo apt update && install nginx libnginx-mod-rtmp
# config file located at `/etc/nginx/nginx.conf`
```

Note: The RTMP version includes all standard Nginx functionality, including load balancing.

## Configuration

The Nginx configuration is managed through a Jinja2 templating system with three key files:

1. `nginx/config.yml` - Contains configuration variables and service definitions
2. `nginx/nginx.conf.j2` - The Jinja2 template file for the Nginx configuration
3. `nginx/nginx.generated.conf` - The final rendered configuration file

### Configuring Services

1. Edit the `config.yml` file to define your services and their configurations:

```yaml
# Example configuration for load balancer
upstreams:
  transcribe_service:
    - name: server-01
      port: 5000
      weight: 3
    - name: server-02
      port: 5000
      weight: 3

# Example configuration for RTMP (only if you need streaming)
rtmp_apps:
  - stream_01
  - stream_02
  - stream_03
```

2. The `nginx.conf.j2` template will use these configurations to generate:
   - Upstream server blocks for load balancing (if upstream services are defined and reachable)
   - Location blocks for service routing (if corresponding upstream server blocks exist)
   - RTMP configuration (if RTMP apps are defined)

### Running the Nginx 
Use the Makefile to render and apply the Nginx configuration:

```bash
# Regular deployment
make nginx

# Deploy with port availability checking
make nginx -check-port=1
```

## Additional Setup for macOS

For macOS users, you may need to configure the firewall:
- Go to **System Preferences** -> **Privacy & Security** and ensure that Nginx is allowed to receive incoming connections

## Troubleshooting
1. **Port conflicts**:
   If you encounter port conflicts, you can clean specific ports:
   ```bash
   # Clean port if it conflicts with 8080
   make clean-ports PORT=8080
   ```
2. **Service management**:
   ```bash
   # Stop nginx services if it already exists
   make stop-nginx
   ```

3. **View Nginx error logs**:
   ```bash
   # On macOS
   tail -f /opt/homebrew/var/log/nginx/error.log

   # On Linux
   tail -f /var/log/nginx/error.log
   ```