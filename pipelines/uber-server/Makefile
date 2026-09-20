.PHONY: all install-redis install-mosquitto install-nginx configure-influxdb-external configure-redis-external configure-mosquitto-external configure-mongodb-external influxdb redis mosquitto mongodb nginx mediamtx flask celery clean-ports stop stop-influxdb stop-redis stop-mosquitto stop-mongodb stop-nginx stop-mediamtx stop-flask stop-celery

# ======= Configurable variables =======
ENV_NAME ?= uber-server
# the TUI passes SUDO="sudo -S" and answers the password prompt itself
SUDO ?= sudo
CONDA_BASE := $(shell conda info --base)
IS_MAC := $(shell uname | grep -qi 'darwin' && echo 1 || echo 0)
BREW_PREFIX := $(shell brew --prefix 2>/dev/null || echo /opt/homebrew)
OPENMMLA_BIND_ADDRESS ?= 0.0.0.0
OPENMMLA_MQTT_ALLOW_ANONYMOUS ?= true
# dashboard backend port; the TUI passes the Dashboard section's value
DASHBOARD_PORT ?= 5050
# MediaMTX stream server (native run; the docker run is in docker/docker-compose.infra.yml)
MEDIAMTX_BIN ?= mediamtx
MEDIAMTX_CONFIG ?= mediamtx/mediamtx.yml
# MediaMTX is started from this folder, so the ./streams/server of mediamtx.yml
# is artifacts/streams/server of the repository: the folder the docker run mounts too
MEDIAMTX_RECORD_ROOT ?= ../../artifacts

MONGO_CONFIG_MAC ?= $(BREW_PREFIX)/etc/mongod.conf
MONGO_CONFIG_LINUX ?= /etc/mongod.conf
REDIS_CONFIG_MAC ?= $(BREW_PREFIX)/etc/redis.conf
REDIS_CONFIG_LINUX ?= /etc/redis/redis.conf
MOSQUITTO_CONFIG_MAC ?= $(BREW_PREFIX)/etc/mosquitto/mosquitto.conf
MOSQUITTO_CONFIG_LINUX ?= /etc/mosquitto/mosquitto.conf
INFLUX_CONFIG_MAC ?= $(BREW_PREFIX)/etc/influxdb/config.toml
INFLUX_CONFIG_LINUX ?= /etc/influxdb/config.toml

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
SERVICES := influxdb redis mosquitto mongodb nginx flask celery
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

# ======= External listener configuration =======
# the TUI passes SUDO="sudo -S", which reads the password from stdin: text
# piped into `sudo tee` would be taken for the password, so a root shell
# writes the text, handed over as an argument. On Linux the config folders
# (/etc/redis) are closed to the login user, so tests and greps use sudo too
sudo_append = $(SUDO) sh -c 'printf "$$1" >> "$$2"' sh $(1) $(2)
sudo_write = $(SUDO) sh -c 'printf "$$1" > "$$2"' sh $(1) $(2)

configure-mongodb-external:
ifeq ($(IS_MAC), 1)
	@config="$(MONGO_CONFIG_MAC)"; \
	if [ -f "$$config" ]; then \
		echo "Configuring MongoDB to listen on $(OPENMMLA_BIND_ADDRESS)"; \
		$(SUDO) cp "$$config" "$$config.openmmla.bak" || true; \
		if grep -qE '^[[:space:]]*bindIp:' "$$config"; then \
			$(SUDO) sed -i '' -E 's/^[[:space:]]*bindIp:.*/  bindIp: $(OPENMMLA_BIND_ADDRESS)/' "$$config"; \
		elif grep -qE '^net:[[:space:]]*$$' "$$config"; then \
			$(SUDO) perl -0pi -e 's/(^net:\n)/$${1}  bindIp: $(OPENMMLA_BIND_ADDRESS)\n/m' "$$config"; \
		else \
			$(call sudo_append,'\nnet:\n  bindIp: $(OPENMMLA_BIND_ADDRESS)\n',"$$config"); \
		fi; \
	else \
		echo "MongoDB config not found at $$config; skipping listener config"; \
	fi
else
	@config="$(MONGO_CONFIG_LINUX)"; \
	if $(SUDO) test -f "$$config"; then \
		echo "Configuring MongoDB to listen on $(OPENMMLA_BIND_ADDRESS)"; \
		$(SUDO) cp "$$config" "$$config.openmmla.bak" || true; \
		if $(SUDO) grep -qE '^[[:space:]]*bindIp:' "$$config"; then \
			$(SUDO) sed -i -E 's/^[[:space:]]*bindIp:.*/  bindIp: $(OPENMMLA_BIND_ADDRESS)/' "$$config"; \
		elif $(SUDO) grep -qE '^net:[[:space:]]*$$' "$$config"; then \
			$(SUDO) perl -0pi -e 's/(^net:\n)/$${1}  bindIp: $(OPENMMLA_BIND_ADDRESS)\n/m' "$$config"; \
		else \
			$(call sudo_append,'\nnet:\n  bindIp: $(OPENMMLA_BIND_ADDRESS)\n',"$$config"); \
		fi; \
	else \
		echo "MongoDB config not found at $$config; skipping listener config"; \
	fi
endif

configure-redis-external:
ifeq ($(IS_MAC), 1)
	@config="$(REDIS_CONFIG_MAC)"; \
	if [ -f "$$config" ]; then \
		echo "Configuring Redis to listen on $(OPENMMLA_BIND_ADDRESS)"; \
		$(SUDO) cp "$$config" "$$config.openmmla.bak" || true; \
		if grep -qE '^bind ' "$$config"; then \
			$(SUDO) sed -i '' -E 's/^bind .*/bind $(OPENMMLA_BIND_ADDRESS) ::/' "$$config"; \
		else \
			$(call sudo_append,'\nbind $(OPENMMLA_BIND_ADDRESS) ::\n',"$$config"); \
		fi; \
		if grep -qE '^protected-mode ' "$$config"; then \
			$(SUDO) sed -i '' -E 's/^protected-mode .*/protected-mode no/' "$$config"; \
		else \
			$(call sudo_append,'protected-mode no\n',"$$config"); \
		fi; \
	else \
		echo "Redis config not found at $$config; skipping listener config"; \
	fi
else
	@config="$(REDIS_CONFIG_LINUX)"; \
	if $(SUDO) test -f "$$config"; then \
		echo "Configuring Redis to listen on $(OPENMMLA_BIND_ADDRESS)"; \
		$(SUDO) cp "$$config" "$$config.openmmla.bak" || true; \
		if $(SUDO) grep -qE '^bind ' "$$config"; then \
			$(SUDO) sed -i -E 's/^bind .*/bind $(OPENMMLA_BIND_ADDRESS) ::/' "$$config"; \
		else \
			$(call sudo_append,'\nbind $(OPENMMLA_BIND_ADDRESS) ::\n',"$$config"); \
		fi; \
		if $(SUDO) grep -qE '^protected-mode ' "$$config"; then \
			$(SUDO) sed -i -E 's/^protected-mode .*/protected-mode no/' "$$config"; \
		else \
			$(call sudo_append,'protected-mode no\n',"$$config"); \
		fi; \
	else \
		echo "Redis config not found at $$config; skipping listener config"; \
	fi
endif

configure-mosquitto-external:
ifeq ($(IS_MAC), 1)
	@config="$(MOSQUITTO_CONFIG_MAC)"; conf_dir="$$(dirname "$$config")/conf.d"; listener_conf="$$conf_dir/openmmla-listener.conf"; \
	if [ -f "$$config" ]; then \
		echo "Configuring Mosquitto to listen on $(OPENMMLA_BIND_ADDRESS)"; \
		$(SUDO) mkdir -p "$$conf_dir"; \
		$(call sudo_write,'listener 1883 $(OPENMMLA_BIND_ADDRESS)\nallow_anonymous $(OPENMMLA_MQTT_ALLOW_ANONYMOUS)\n',"$$listener_conf"); \
		if ! grep -qE "^[[:space:]]*include_dir[[:space:]]+$$conf_dir" "$$config"; then \
			$(call sudo_append,"\ninclude_dir $$conf_dir\n","$$config"); \
		fi; \
	else \
		echo "Mosquitto config not found at $$config; skipping listener config"; \
	fi
else
	@config="$(MOSQUITTO_CONFIG_LINUX)"; conf_dir="/etc/mosquitto/conf.d"; listener_conf="$$conf_dir/openmmla-listener.conf"; \
	if $(SUDO) test -f "$$config" || $(SUDO) test -d "$$conf_dir"; then \
		echo "Configuring Mosquitto to listen on $(OPENMMLA_BIND_ADDRESS)"; \
		$(SUDO) mkdir -p "$$conf_dir"; \
		$(call sudo_write,'listener 1883 $(OPENMMLA_BIND_ADDRESS)\nallow_anonymous $(OPENMMLA_MQTT_ALLOW_ANONYMOUS)\n',"$$listener_conf"); \
	else \
		echo "Mosquitto config not found; skipping listener config"; \
	fi
endif

configure-influxdb-external:
ifeq ($(IS_MAC), 1)
	@config="$(INFLUX_CONFIG_MAC)"; \
	if [ -f "$$config" ]; then \
		echo "Configuring InfluxDB to listen on all interfaces"; \
		$(SUDO) cp "$$config" "$$config.openmmla.bak" || true; \
		if grep -qE '^[#[:space:]]*http-bind-address[[:space:]]*=' "$$config"; then \
			$(SUDO) sed -i '' -E 's|^[#[:space:]]*http-bind-address[[:space:]]*=.*|http-bind-address = ":8086"|' "$$config"; \
		else \
			$(call sudo_append,'\nhttp-bind-address = ":8086"\n',"$$config"); \
		fi; \
	else \
		echo "InfluxDB config not found at $$config; default InfluxDB bind is usually :8086"; \
	fi
else
	@config="$(INFLUX_CONFIG_LINUX)"; \
	if $(SUDO) test -f "$$config"; then \
		echo "Configuring InfluxDB to listen on all interfaces"; \
		$(SUDO) cp "$$config" "$$config.openmmla.bak" || true; \
		if $(SUDO) grep -qE '^[#[:space:]]*http-bind-address[[:space:]]*=' "$$config"; then \
			$(SUDO) sed -i -E 's|^[#[:space:]]*http-bind-address[[:space:]]*=.*|http-bind-address = ":8086"|' "$$config"; \
		else \
			$(call sudo_append,'\nhttp-bind-address = ":8086"\n',"$$config"); \
		fi; \
	else \
		echo "InfluxDB config not found at $$config; default InfluxDB bind is usually :8086"; \
	fi
endif

# ======= Individual services =======
influxdb: configure-influxdb-external
ifeq ($(IS_MAC), 1)
	@echo "🔁 Restarting InfluxDB (macOS)"
	brew services restart influxdb
else
	@echo "🔁 Restarting InfluxDB (Linux)"
	$(SUDO) systemctl enable influxdb
	$(SUDO) systemctl restart influxdb
endif

# ======= Install (native) =======
# `make <service>` (the Start of a card) installs the package first when this
# host has none: brew on macOS, apt on Debian and Ubuntu. make runs on the
# card's host, so IS_MAC tells that host's platform. InfluxDB and MongoDB are
# not installed here: natively they need a vendor apt repository
# (docs/system_services.md), and by default they run from the docker stack.
define apt_install
{ command -v apt-get >/dev/null 2>&1 || { echo "❌ $(1) is not installed and this is not an apt system: install $(2) with the package manager here, then run this again"; exit 1; }; echo "📦 Installing $(1) (apt)"; $(SUDO) apt-get update -qq && $(SUDO) env DEBIAN_FRONTEND=noninteractive apt-get install -y $(2); }
endef

install-redis:
ifeq ($(IS_MAC), 1)
	@command -v redis-server >/dev/null 2>&1 || { echo "📦 Installing Redis (brew)"; brew install redis; }
else
	@command -v redis-server >/dev/null 2>&1 || $(call apt_install,Redis,redis-server)
endif

install-mosquitto:
ifeq ($(IS_MAC), 1)
	@command -v mosquitto >/dev/null 2>&1 || { echo "📦 Installing Mosquitto (brew)"; brew install mosquitto; }
else
	@command -v mosquitto >/dev/null 2>&1 || $(call apt_install,Mosquitto,mosquitto mosquitto-clients)
endif

install-nginx:
ifeq ($(IS_MAC), 1)
	@command -v nginx >/dev/null 2>&1 || { echo "📦 Installing Nginx (brew)"; brew install nginx; }
else
	@command -v nginx >/dev/null 2>&1 || $(call apt_install,Nginx,nginx)
endif

redis: install-redis configure-redis-external
ifeq ($(IS_MAC), 1)
	@echo "🔁 Restarting Redis (macOS)"
	brew services restart redis
else
	@echo "🔁 Restarting Redis (Linux)"
	$(SUDO) systemctl enable redis-server
	$(SUDO) systemctl restart redis-server
endif

mosquitto: install-mosquitto configure-mosquitto-external
ifeq ($(IS_MAC), 1)
	@echo "🔁 Restarting Mosquitto (macOS)"
	brew services restart mosquitto
else
	@echo "🔁 Restarting Mosquitto (Linux)"
	$(SUDO) systemctl enable mosquitto
	$(SUDO) systemctl restart mosquitto
endif

mongodb: configure-mongodb-external
ifeq ($(IS_MAC), 1)
	@echo "🔁 Restarting MongoDB (macOS)"
	brew services restart mongodb-community
else
	@echo "🔁 Restarting MongoDB (Linux)"
	$(SUDO) systemctl enable mongod
	$(SUDO) systemctl restart mongod
endif

# renders the config, installs it and reloads nginx (starting it if it is not
# running): the old workers finish the requests they hold, so a session in
# progress keeps its answers, and the reload clears what nginx learned of the
# servers, so one that just started is tried at once instead of after fail_timeout
nginx: install-nginx
	@echo "🔁 Rendering and reloading NGINX"
	@test -f nginx/config.yml || { echo "❌ nginx/config.yml is missing: fill in the Config tab of the Gateway card in the console, or copy nginx/config_template.yml to it and edit it"; exit 1; }
ifeq ($(check_port), 1)
	$(CONDA_ACTIVATE) && python3 nginx/render_nginx.py nginx/config.yml nginx/nginx.conf.j2 nginx/nginx.generated.conf --port-check
else
	$(CONDA_ACTIVATE) && python3 nginx/render_nginx.py nginx/config.yml nginx/nginx.conf.j2 nginx/nginx.generated.conf
endif
ifeq ($(IS_MAC), 1)
	@$(SUDO) sh -c 'conf=$(BREW_PREFIX)/etc/nginx/nginx.conf; \
		if [ -f "$$conf" ]; then cp "$$conf" "$$conf.openmmla.bak"; fi; \
		cp nginx/nginx.generated.conf "$$conf"; \
		if ! nginx -t; then \
			echo "❌ the rendered config does not pass nginx -t: keeping the one that ran"; \
			if [ -f "$$conf.openmmla.bak" ]; then cp "$$conf.openmmla.bak" "$$conf"; fi; \
			exit 1; \
		fi; \
		if pgrep nginx > /dev/null; then \
			echo "🔁 Reloading NGINX (macOS)"; nginx -s reload; \
		else \
			echo "🚀 Starting NGINX (macOS)"; nginx; \
		fi'
else
	@$(SUDO) sh -c 'conf=/etc/nginx/nginx.conf; \
		if [ -f "$$conf" ]; then cp "$$conf" "$$conf.openmmla.bak"; fi; \
		cp nginx/nginx.generated.conf "$$conf"; \
		if ! nginx -t; then \
			echo "❌ the rendered config does not pass nginx -t: keeping the one that ran"; \
			if [ -f "$$conf.openmmla.bak" ]; then cp "$$conf.openmmla.bak" "$$conf"; fi; \
			exit 1; \
		fi; \
		echo "🔁 Reloading NGINX (Linux)"; \
		systemctl enable nginx; systemctl reload-or-restart nginx'
endif

flask:
	@echo "🔁 Starting Flask"
	tmux new-session -d -s flask -n flask bash
	tmux send-keys -t flask 'cd dashboard/flask-backend' C-m
	tmux send-keys -t flask 'source $(CONDA_BASE)/etc/profile.d/conda.sh && conda activate $(ENV_NAME)' C-m
	tmux send-keys -t flask 'gunicorn -k gevent -w 1 -b 0.0.0.0:$(DASHBOARD_PORT) dashboard:app' C-m

celery:
	@echo "🔁 Starting Celery worker"
	tmux new-session -d -s celery -n celery bash
	tmux send-keys -t celery 'cd dashboard/flask-backend' C-m
	tmux send-keys -t celery 'source $(CONDA_BASE)/etc/profile.d/conda.sh && conda activate $(ENV_NAME)' C-m
	tmux send-keys -t celery 'celery -A dashboard.celery worker --loglevel=info' C-m

mediamtx:
	@echo "🔁 Starting MediaMTX"
	@command -v $(MEDIAMTX_BIN) >/dev/null 2>&1 || { echo "❌ $(MEDIAMTX_BIN) not found: brew install mediamtx, or unpack a release into /usr/local/bin (docs/rtmp_streaming.md)"; exit 1; }
	@mkdir -p '$(abspath $(MEDIAMTX_RECORD_ROOT))/streams/server'
	tmux new-session -d -s mediamtx -n mediamtx -c '$(abspath $(MEDIAMTX_RECORD_ROOT))' bash
	tmux send-keys -t mediamtx '$(MEDIAMTX_BIN) $(abspath $(MEDIAMTX_CONFIG))' C-m

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
	@$(SUDO) sh -c 'launchctl bootout system/homebrew.mxcl.nginx-full 2>/dev/null; \
		nginx -s stop 2>/dev/null; \
		pkill -f "nginx: master" 2>/dev/null; \
		true'
else
	@echo "🛑 Stopping NGINX (Linux)"
	@$(SUDO) sh -c 'systemctl stop nginx || echo "⚠️ NGINX not running"; \
		systemctl disable nginx || echo "⚠️ Could not disable NGINX"'
endif

stop-mediamtx:
	@echo "🛑 Stopping MediaMTX"
	@if tmux has-session -t mediamtx 2>/dev/null; then \
		tmux send-keys -t mediamtx C-c; \
		sleep 1; \
		tmux kill-session -t mediamtx; \
	else \
		echo "⚠️ No MediaMTX session found"; \
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
