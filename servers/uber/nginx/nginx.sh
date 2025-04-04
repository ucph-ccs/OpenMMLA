  #!/bin/bash

  BASH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  CONFIG_YAML="$BASH_DIR/config.yml"
  TEMPLATE_J2="$BASH_DIR/nginx.conf.j2"
  OUTPUT_CONF="$BASH_DIR/nginx.generated.conf"

  # 判断 OS 类型
  if [[ "$OSTYPE" == "darwin"* ]]; then
      OS_TYPE="macos"
      NGINX_CONF="/opt/homebrew/etc/nginx/nginx.conf"
  else
      OS_TYPE="linux"
      NGINX_CONF="/etc/nginx/nginx.conf"
  fi

  echo "NGINX config path: $NGINX_CONF"

  # 判断是否启用端口检查
  PORT_CHECK_FLAG=""
  if [[ "$1" == "--port-check" ]]; then
      PORT_CHECK_FLAG="--port-check"
      echo "🔍 Port connectivity check is ENABLED."
  else
      echo "🧊 Port connectivity check is DISABLED (default)."
  fi

  # 渲染配置
  python3 "$BASH_DIR/render_nginx.py" "$CONFIG_YAML" "$TEMPLATE_J2" "$OUTPUT_CONF" $PORT_CHECK_FLAG

  # 拷贝配置到 NGINX 路径
  sudo cp "$OUTPUT_CONF" "$NGINX_CONF"

  # 检查 nginx 是否在运行
  if pgrep nginx > /dev/null; then
      echo "🔁 Reloading NGINX..."
      sudo nginx -t && sudo nginx -s reload
  else
      echo "🚀 Starting NGINX..."
      sudo nginx
  fi
