.PHONY: all influxdb redis mosquitto nginx flask next celery clean-ports stop stop-influxdb stop-redis stop-mosquitto stop-nginx stop-next stop-flask stop-celery

# ======= Configurable variables =======
ENV_NAME ?= uber-server
IS_MAC := $(shell uname | grep -qi 'darwin' && echo 1 || echo 0)
-port ?= 
-check-port ?= 
-without ?= 
CONDA_PREFIX := $(shell conda info --base)
CONDA_ACTIVATE := . $(CONDA_PREFIX)/etc/profile.d/conda.sh && conda activate $(ENV_NAME)

# ======= Service exclusion ========
SERVICES := influxdb redis mosquitto nginx flask next celery
FILTERED_SERVICES := $(filter-out $(subst $(comma),$(space),$(-without)),$(SERVICES))

# Helper variables for parsing
comma := ,
space :=
space +=

# ======= Default task =======
all: sudo-keepalive
	@for service in $(FILTERED_SERVICES); do \
		$(MAKE) clean-ports stop-$$service $$service; \
	done

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
ifeq ($(-check-port), 1)
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
	@echo "🧹 Cleaning ports..."
	@if [ -n "$(-port)" ]; then \
		echo "Cleaning port $(-port)..."; \
		sudo lsof -ti:$(-port) | xargs -r sudo kill -9 || true; \
	else \
		echo "Cleaning default ports for services: $(FILTERED_SERVICES)"; \
		for p in $(filter 8086,$(if $(filter influxdb,$(FILTERED_SERVICES)),8086,)) \
				 $(filter 6379,$(if $(filter redis,$(FILTERED_SERVICES)),6379,)) \
				 $(filter 1883,$(if $(filter mosquitto,$(FILTERED_SERVICES)),1883,)) \
				 $(filter 8080,$(if $(filter nginx,$(FILTERED_SERVICES)),8080,)) \
				 $(filter 3000,$(if $(filter next,$(FILTERED_SERVICES)),3000,)) \
				 $(filter 5000,$(if $(filter flask,$(FILTERED_SERVICES)),5000,)); do \
			echo "  Cleaning port $$p..."; \
			sudo lsof -ti:$$p | xargs -r sudo kill -9 || true; \
		done; \
	fi

# ======= Individual stop functions =======
stop-influxdb:
ifeq ($(IS_MAC), 1)
	@echo "🛑 Stopping InfluxDB (macOS)"
	@brew services stop influxdb 2>/dev/null || echo "⚠️ InfluxDB not running"
else
	@echo "🛑 Stopping InfluxDB (Linux)"
	@sudo systemctl stop influxdb || echo "⚠️ InfluxDB not running"
endif

stop-redis:
ifeq ($(IS_MAC), 1)
	@echo "🛑 Stopping Redis (macOS)"
	@brew services stop redis 2>/dev/null || echo "⚠️ Redis not running"
else
	@echo "🛑 Stopping Redis (Linux)"
	@sudo systemctl stop redis-server || echo "⚠️ Redis not running"
endif

stop-mosquitto:
ifeq ($(IS_MAC), 1)
	@echo "🛑 Stopping Mosquitto (macOS)"
	@brew services stop mosquitto 2>/dev/null || echo "⚠️ Mosquitto not running"
else
	@echo "🛑 Stopping Mosquitto (Linux)"
	@sudo systemctl stop mosquitto || echo "⚠️ Mosquitto not running"
endif

stop-nginx:
ifeq ($(IS_MAC), 1)
	@echo "🛑 Stopping NGINX (macOS)"
	@sudo nginx -s stop 2>/dev/null || echo "⚠️ NGINX not running"
else
	@echo "🛑 Stopping NGINX (Linux)"
	@sudo systemctl stop nginx || echo "⚠️ NGINX not running"
endif

stop-next:
	@echo "🛑 Stopping Next.js frontend"
	@tmux has-session -t next 2>/dev/null && tmux kill-session -t next && echo "Killed tmux session: next" || echo "⚠️ No Next.js session found"

stop-flask:
	@echo "🛑 Stopping Flask backend"
	@tmux has-session -t flask 2>/dev/null && tmux kill-session -t flask && echo "Killed tmux session: flask" || echo "⚠️ No Flask session found"

stop-celery:
	@echo "🛑 Stopping Celery worker"
	@tmux has-session -t celery 2>/dev/null && tmux kill-session -t celery && echo "Killed tmux session: celery" || echo "⚠️ No Celery session found"

stop:
	@echo "🛑 Stopping services: $(FILTERED_SERVICES)"
	@for service in $(FILTERED_SERVICES); do \
		$(MAKE) stop-$$service; \
	done
	@echo "✅ Services stopped."