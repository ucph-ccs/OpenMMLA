.PHONY: all influxdb redis mosquitto mongodb nginx flask next celery clean-ports stop stop-influxdb stop-redis stop-mosquitto stop-mongodb stop-nginx stop-next stop-flask stop-celery

# ======= Configurable variables =======
ENV_NAME ?= uber-server
SUDO ?= sudo
CONDA_BASE := $(shell conda info --base)
IS_MAC := $(shell uname | grep -qi 'darwin' && echo 1 || echo 0)

# extract command and arguments
CMD := $(firstword $(MAKECMDGOALS))
ARGS := $(wordlist 2, 99, $(MAKECMDGOALS))

# clean-ports，assign ARGS to PORTS
ports := $(if $(filter clean-ports,$(CMD)),$(ARGS),)
check_port ?= $(if $(filter nginx,$(CMD)),$(if $(filter false,$(ARGS)),0,1),1)
without ?=

CONDA_PREFIX := $(shell conda info --base)
CONDA_ACTIVATE := . $(CONDA_PREFIX)/etc/profile.d/conda.sh && conda activate $(ENV_NAME)

# ======= Service exclusion ========
SERVICES := influxdb redis mosquitto mongodb nginx flask next celery
FILTERED_SERVICES = $(filter-out $(subst $(comma),$(space),$(without)),$(SERVICES))

# Helper variables for parsing
comma := ,
space :=
space +=


# ======= Default task =======
all: sudo-keepalive clean-ports
	@for service in $(FILTERED_SERVICES); do \
		$(MAKE) stop-$$service $$service; \
	done

# ======= Autostart services =======
sudo-keepalive:
	@echo "🔐 Ensuring sudo session stays alive..."
	@$(SUDO) -v
	@while true; do $(SUDO) -n true; sleep 60; done 2>/dev/null &

# ======= Individual services =======
influxdb:
ifeq ($(IS_MAC), 1)
	@echo "🔁 Restarting InfluxDB (macOS)"
	brew services restart influxdb
else
	@echo "🔁 Restarting InfluxDB (Linux)"
	$(SUDO) systemctl enable influxdb
	$(SUDO) systemctl restart influxdb
endif

redis:
ifeq ($(IS_MAC), 1)
	@echo "🔁 Restarting Redis (macOS)"
	brew services restart redis
else
	@echo "🔁 Restarting Redis (Linux)"
	$(SUDO) systemctl enable redis-server
	$(SUDO) systemctl restart redis-server
endif

mosquitto:
ifeq ($(IS_MAC), 1)
	@echo "🔁 Restarting Mosquitto (macOS)"
	brew services restart mosquitto
else
	@echo "🔁 Restarting Mosquitto (Linux)"
	$(SUDO) systemctl enable mosquitto
	$(SUDO) systemctl restart mosquitto
endif

mongodb:
ifeq ($(IS_MAC), 1)
	@echo "🔁 Restarting MongoDB (macOS)"
	brew services restart mongodb-community
else
	@echo "🔁 Restarting MongoDB (Linux)"
	$(SUDO) systemctl enable mongod
	$(SUDO) systemctl restart mongod
endif

nginx:
	@echo "🔁 Rendering and starting NGINX"
ifeq ($(check_port), 1)
	$(CONDA_ACTIVATE) && python3 nginx/render_nginx.py nginx/config.yml nginx/nginx.conf.j2 nginx/nginx.generated.conf --port-check
else
	$(CONDA_ACTIVATE) && python3 nginx/render_nginx.py nginx/config.yml nginx/nginx.conf.j2 nginx/nginx.generated.conf
endif
ifeq ($(IS_MAC), 1)
	$(SUDO) cp nginx/nginx.generated.conf /opt/homebrew/etc/nginx/nginx.conf
else
	$(SUDO) cp nginx/nginx.generated.conf /etc/nginx/nginx.conf
endif
ifeq ($(IS_MAC), 1)
	@if pgrep nginx > /dev/null; then \
		echo "🔁 Reloading NGINX (macOS)"; \
		$(SUDO) nginx -t && $(SUDO) nginx -s reload; \
	else \
		echo "🚀 Starting NGINX (macOS)"; \
		$(SUDO) nginx; \
	fi
else
	echo "🔁 Restarting NGINX (Linux)"; \
	$(SUDO) systemctl enable nginx
	$(SUDO) systemctl restart nginx
endif

next:
	@echo "🔁 Starting Next.js frontend"
	tmux new-session -d -s next 'cd dashboard/next-frontend && npm run build && npm run start'

flask:
	@echo "🔁 Starting Flask"
	tmux new-session -d -s flask -n flask bash
	tmux send-keys -t flask 'cd dashboard/flask-backend' C-m
	tmux send-keys -t flask 'source $(CONDA_BASE)/etc/profile.d/conda.sh && conda activate $(ENV_NAME)' C-m
	tmux send-keys -t flask 'gunicorn -k gevent -w 1 -b 0.0.0.0:5050 dashboard:app' C-m

celery:
	@echo "🔁 Starting Celery worker"
	tmux new-session -d -s celery -n celery bash
	tmux send-keys -t celery 'cd dashboard/flask-backend' C-m
	tmux send-keys -t celery 'source $(CONDA_BASE)/etc/profile.d/conda.sh && conda activate $(ENV_NAME)' C-m
	tmux send-keys -t celery 'celery -A dashboard.celery worker --loglevel=info' C-m

# ======= Utility =======
clean-ports:
	@echo "🧹 Cleaning ports..."
	@if [ -n "$(ports)" ]; then \
		for p in $(ports); do \
			echo "  Cleaning port $$p..."; \
			$(SUDO) lsof -ti:$$p | xargs -r $(SUDO) kill -9 || true; \
		done; \
	else \
		echo "Cleaning default ports for services: $(FILTERED_SERVICES)"; \
		for p in $(filter 8086,$(if $(filter influxdb,$(FILTERED_SERVICES)),8086,)) \
				 $(filter 6379,$(if $(filter redis,$(FILTERED_SERVICES)),6379,)) \
				 $(filter 1883,$(if $(filter mosquitto,$(FILTERED_SERVICES)),1883,)) \
				 $(filter 27017,$(if $(filter mongodb,$(FILTERED_SERVICES)),27017,)) \
				 $(filter 8080,$(if $(filter nginx,$(FILTERED_SERVICES)),8080,)) \
				 $(filter 3000,$(if $(filter next,$(FILTERED_SERVICES)),3000,)) \
				 $(filter 5050,$(if $(filter flask,$(FILTERED_SERVICES)),5050,)); do \
			echo "  Cleaning port $$p..."; \
			$(SUDO) lsof -ti:$$p | xargs -r $(SUDO) kill -9 || true; \
		done; \
	fi


# ======= Individual stop functions =======
stop-influxdb:
ifeq ($(IS_MAC), 1)
	@echo "🛑 Stopping InfluxDB (macOS)"
	@brew services stop influxdb 2>/dev/null || echo "⚠️ InfluxDB not running"
else
	@echo "🛑 Stopping InfluxDB (Linux)"
	@$(SUDO) systemctl stop influxdb || echo "⚠️ InfluxDB not running"
	@$(SUDO) systemctl disable influxdb || echo "⚠️ Could not disable InfluxDB"
endif

stop-redis:
ifeq ($(IS_MAC), 1)
	@echo "🛑 Stopping Redis (macOS)"
	@brew services stop redis 2>/dev/null || echo "⚠️ Redis not running"
else
	@echo "🛑 Stopping Redis (Linux)"
	@$(SUDO) systemctl stop redis-server || echo "⚠️ Redis not running"
	@$(SUDO) systemctl disable redis-server || echo "⚠️ Could not disable Redis"
endif

stop-mosquitto:
ifeq ($(IS_MAC), 1)
	@echo "🛑 Stopping Mosquitto (macOS)"
	@brew services stop mosquitto 2>/dev/null || echo "⚠️ Mosquitto not running"
else
	@echo "🛑 Stopping Mosquitto (Linux)"
	@$(SUDO) systemctl stop mosquitto || echo "⚠️ Mosquitto not running"
	@$(SUDO) systemctl disable mosquitto || echo "⚠️ Could not disable Mosquitto"
endif

stop-mongodb:
ifeq ($(IS_MAC), 1)
	@echo "🛑 Stopping MongoDB (macOS)"
	@brew services stop mongodb-community 2>/dev/null || echo "⚠️ MongoDB not running"
	@pkill -f mongod 2>/dev/null || true
else
	@echo "🛑 Stopping MongoDB (Linux)"
	@$(SUDO) systemctl stop mongod || echo "⚠️ MongoDB not running"
	@$(SUDO) systemctl disable mongod || echo "⚠️ Could not disable MongoDB"
endif

stop-nginx:
ifeq ($(IS_MAC), 1)
	@echo "🛑 Stopping NGINX (macOS)"
	@$(SUDO) launchctl bootout system/homebrew.mxcl.nginx-full 2>/dev/null || true
	@$(SUDO) nginx -s stop || true
	@$(SUDO) pkill -f "nginx: master" || true
else
	@echo "🛑 Stopping NGINX (Linux)"
	@$(SUDO) systemctl stop nginx || echo "⚠️ NGINX not running"
	@$(SUDO) systemctl disable nginx || echo "⚠️ Could not disable NGINX"
endif

stop-next:
	@echo "🛑 Stopping Next.js frontend"
	@if tmux has-session -t next 2>/dev/null; then \
		echo "Sending Ctrl+C to Next.js session..."; \
		tmux send-keys -t next C-c; \
		sleep 1; \
		tmux kill-session -t next; \
	else \
		echo "⚠️ No Next.js session found"; \
	fi

stop-flask:
	@echo "🛑 Stopping Flask backend"
	@if tmux has-session -t flask 2>/dev/null; then \
		echo "Sending Ctrl+C to Flask session..."; \
		tmux send-keys -t flask C-c; \
		sleep 1; \
		tmux kill-session -t flask; \
	else \
		echo "⚠️ No Flask session found"; \
	fi

stop-celery:
	@echo "🛑 Stopping Celery worker"
	@if tmux has-session -t celery 2>/dev/null; then \
		echo "Sending Ctrl+C to Celery session..."; \
		tmux send-keys -t celery C-c; \
		sleep 1; \
		tmux kill-session -t celery; \
	else \
		echo "⚠️ No Celery session found"; \
	fi

stop:
	@echo "🛑 Stopping services: $(FILTERED_SERVICES)"
	@for service in $(FILTERED_SERVICES); do \
		$(MAKE) stop-$$service; \
	done
	@echo "✅ Services stopped."

%:
	@: