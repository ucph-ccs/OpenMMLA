.PHONY: all influxdb redis mosquitto nginx flask next celery clean-ports stop

# ======= Configurable variables =======
ENV_NAME ?= uber-server
IS_MAC := $(shell uname | grep -qi 'darwin' && echo 1 || echo 0)
PORT := $(or $(PORT), $(port))
CONDA_PREFIX := $(shell conda info --base)
CONDA_ACTIVATE := . $(CONDA_PREFIX)/etc/profile.d/conda.sh && conda activate $(ENV_NAME)

# ======= Default task =======
all: sudo-keepalive clean-ports stop influxdb redis mosquitto nginx next flask celery

# ======= Autostart services =======
sudo-keepalive:
	@echo "🔐 Ensuring sudo session stays alive..."
	@sudo -v
	@while true; do sudo -n true; sleep 60; done 2>/dev/null &

# ======= Individual services =======
influxdb:
ifeq ($(IS_MAC), 1)
	@echo "🔁 Restarting InfluxDB (macOS)"
	brew services restart influxdb
else
	@echo "🔁 Restarting InfluxDB (Linux)"
	sudo systemctl restart influxdb
endif

redis:
ifeq ($(IS_MAC), 1)
	@echo "🔁 Restarting Redis (macOS)"
	brew services restart redis
else
	@echo "🔁 Restarting Redis (Linux)"
	sudo systemctl restart redis-server
endif

mosquitto:
ifeq ($(IS_MAC), 1)
	@echo "🔁 Restarting Mosquitto (macOS)"
	brew services restart mosquitto
else
	@echo "🔁 Restarting Mosquitto (Linux)"
	sudo systemctl restart mosquitto
endif

nginx:
	@echo "🔁 Rendering and starting NGINX"
ifeq ($(PORT), 1)
	$(CONDA_ACTIVATE) && python3 nginx/render_nginx.py nginx/config.yml nginx/nginx.conf.j2 nginx/nginx.generated.conf --port-check
else
	$(CONDA_ACTIVATE) && python3 nginx/render_nginx.py nginx/config.yml nginx/nginx.conf.j2 nginx/nginx.generated.conf
endif
ifeq ($(IS_MAC), 1)
	sudo cp nginx/nginx.generated.conf /opt/homebrew/etc/nginx/nginx.conf
else
	sudo cp nginx/nginx.generated.conf /etc/nginx/nginx.conf
endif
ifeq ($(IS_MAC), 1)
	@if pgrep nginx > /dev/null; then \
		echo "🔁 Reloading NGINX (macOS)"; \
		sudo nginx -t && sudo nginx -s reload; \
	else \
		echo "🚀 Starting NGINX (macOS)"; \
		sudo nginx; \
	fi
else
	echo "🔁 Reloading NGINX (Linux)"; \
	sudo systemctl restart nginx
endif

next:
	@echo "🔁 Starting Next.js frontend"
	tmux new-session -d -s next 'cd dashboard/next-frontend && npm run build && npm run start'

flask:
	@echo "🔁 Starting Flask"
	tmux new-session -d -s flask -n flask bash
	tmux send-keys -t flask 'cd dashboard/flask-backend' C-m
	tmux send-keys -t flask 'source activate $(ENV_NAME)' C-m
	tmux send-keys -t flask 'gunicorn -k gevent -w 1 -b 0.0.0.0:5000 dashboard:app' C-m

celery:
	@echo "🔁 Starting Celery worker"
	tmux new-session -d -s celery -n celery bash
	tmux send-keys -t celery 'cd dashboard/flask-backend' C-m
	tmux send-keys -t celery 'source activate $(ENV_NAME)' C-m
	tmux send-keys -t celery 'celery -A dashboard.celery worker --loglevel=info' C-m

# ======= Utility =======
clean-ports:
	@echo "🧹 Cleaning all reserved ports..."
	@for p in 8086 6379 1883 8080 3000 5000; do \
		sudo lsof -ti:$$p | xargs -r sudo kill -9 || true; \
	done

stop:
ifeq ($(IS_MAC), 1)
	@echo "🛑 Stopping services (macOS)"
	@sudo nginx -s stop 2>/dev/null || echo "⚠️ NGINX not running"
	@brew services stop influxdb 2>/dev/null || echo "⚠️ InfluxDB not running"
	@brew services stop redis 2>/dev/null || echo "⚠️ Redis not running"
	@brew services stop mosquitto 2>/dev/null || echo "⚠️ Mosquitto not running"
else
	@echo "🛑 Stopping services (Linux)"
	@sudo systemctl stop nginx || echo "⚠️ NGINX not running"
	@sudo systemctl stop influxdb || echo "⚠️ InfluxDB not running"
	@sudo systemctl stop redis || echo "⚠️ Redis not running"
	@sudo systemctl stop mosquitto || echo "⚠️ Mosquitto not running"
endif
	@echo "🛑 Stopping tmux-based services"
	@for name in flask next celery; do \
		tmux has-session -t $$name 2>/dev/null && tmux kill-session -t $$name && echo "Killed tmux session: $$name" || echo "No session found: $$name"; \
	done
	@echo "✅ All services stopped."