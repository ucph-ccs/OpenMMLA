.PHONY: all install-redis install-mosquitto install-nginx configure-influxdb-external configure-redis-external configure-mosquitto-external configure-mongodb-external influxdb redis mosquitto mongodb nginx mediamtx flask celery autostart no-autostart boot wait-for-bind infra-after-boot clean-ports stop stop-influxdb stop-redis stop-mosquitto stop-mongodb stop-nginx stop-mediamtx stop-flask stop-celery

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
# the dashboard's media ports: the same app on more ports, bound on the same addresses, from which
# the Live page loads the recorded videos and sound of a replay. A browser opens at most six HTTP/1.1
# connections per origin and every file it plays holds one, so from ports of their own they leave the
# page's requests and live stream theirs, and two ports carry six videos and the sound at once. The
# two ports above DASHBOARD_PORT unless set (spaces or commas between them); DASHBOARD_MEDIA_PORT
# names one port instead; empty or 0 opens none (the page then loads the files from DASHBOARD_PORT,
# fewer at once)
# each word once, in order (comma and space are the parsing helpers further down)
uniq = $(if $1,$(firstword $1) $(call uniq,$(filter-out $(firstword $1),$1)))
DASHBOARD_MEDIA_PORTS ?= $(if $(filter undefined,$(origin DASHBOARD_MEDIA_PORT)),$(shell expr '$(DASHBOARD_PORT)' + 1 2>/dev/null) $(shell expr '$(DASHBOARD_PORT)' + 2 2>/dev/null),$(DASHBOARD_MEDIA_PORT))
DASHBOARD_MEDIA_LISTEN = $(strip $(call uniq,$(filter-out 0 $(DASHBOARD_PORT),$(subst $(comma),$(space),$(DASHBOARD_MEDIA_PORTS)))))
# where the dashboard listens: where the docker stack publishes InfluxDB, MongoDB and MediaMTX
# (INFRA_BIND_ADDRESS and INFRA_LOOPBACK_ADDRESS of docker/.env), so a host that keeps them on its
# tailnet address keeps the dashboard, its transcripts and recordings off its other networks too;
# every interface when docker/.env names none. Several addresses are space-separated
DOCKER_ENV_FILE := $(abspath ../../docker/.env)
INFRA_BIND_ADDRESS ?= $(shell sed -n 's/^INFRA_BIND_ADDRESS=//p' '$(DOCKER_ENV_FILE)' 2>/dev/null | tail -n 1)
INFRA_LOOPBACK_ADDRESS ?= $(shell sed -n 's/^INFRA_LOOPBACK_ADDRESS=//p' '$(DOCKER_ENV_FILE)' 2>/dev/null | tail -n 1)
DASHBOARD_BIND ?= $(or $(strip $(INFRA_BIND_ADDRESS) $(INFRA_LOOPBACK_ADDRESS)),$(OPENMMLA_BIND_ADDRESS))
# the dashboard's web process and worker run through restart_on_failure.sh, which starts one again
# when it exits with an error (three times in a row at most), and what their tmux panes show is
# copied into logs/flask.log and logs/celery.log, which keep what a crashed process said after its
# pane is gone. A log past LOG_ROTATE_BYTES becomes <name>.log.1 at the next start
LOG_DIR := $(CURDIR)/logs
LOG_ROTATE_BYTES ?= 10485760
RESTART_ON_FAILURE := bash "$(CURDIR)/restart_on_failure.sh"
# makes logs/ and moves logs/<name>.log to <name>.log.1 once it is past LOG_ROTATE_BYTES, unless the
# tmux session of that name still runs (its pane still writes into the file)
rotate_log = mkdir -p '$(LOG_DIR)' && f='$(LOG_DIR)/$(1).log' && if ! tmux has-session -t '=$(1)' 2>/dev/null && [ -f "$$f" ] && [ $$(( $$(wc -c < "$$f") )) -gt $(LOG_ROTATE_BYTES) ]; then mv -f "$$f" "$$f.1"; fi
# after a reboot (make autostart): the targets that start, and how long to wait for their addresses
AUTOSTART ?= flask celery
BOOT_WAIT_SECONDS ?= 300
CRONTAB ?= crontab
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

# Helper variables for parsing. A space is an empty variable on each side of one: `space :=` then
# `space +=` gave one under macOS make 3.81 but nothing under GNU Make 4.3 (Ubuntu), which turned
# without=nginx,celery into nginxcelery and excluded nothing
comma := ,
empty :=
space := $(empty) $(empty)


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
	@$(call rotate_log,flask)
	tmux new-session -d -s flask -n flask bash
	tmux pipe-pane -t flask "cat >> '$(LOG_DIR)/flask.log'"
	tmux send-keys -t flask 'cd dashboard/flask-backend' C-m
	tmux send-keys -t flask 'source $(CONDA_BASE)/etc/profile.d/conda.sh && conda activate $(ENV_NAME)' C-m
	tmux send-keys -t flask 'DASHBOARD_MEDIA_PORTS="$(DASHBOARD_MEDIA_LISTEN)" $(RESTART_ON_FAILURE) gunicorn -k gevent -w 1 $(foreach address,$(DASHBOARD_BIND),$(strip -b $(address):$(DASHBOARD_PORT) $(foreach port,$(DASHBOARD_MEDIA_LISTEN),-b $(address):$(port)))) dashboard:app' C-m

celery:
	@echo "🔁 Starting Celery worker"
	@$(call rotate_log,celery)
	tmux new-session -d -s celery -n celery bash
	tmux pipe-pane -t celery "cat >> '$(LOG_DIR)/celery.log'"
	tmux send-keys -t celery 'cd dashboard/flask-backend' C-m
	tmux send-keys -t celery 'source $(CONDA_BASE)/etc/profile.d/conda.sh && conda activate $(ENV_NAME)' C-m
	tmux send-keys -t celery '$(RESTART_ON_FAILURE) celery -A dashboard.celery worker --loglevel=info' C-m

# ======= Start the dashboard after a reboot =======
# `make autostart DASHBOARD_PORT=<port>` adds one line to the login user's crontab that runs
# `make boot` in this folder after every reboot, and `make no-autostart` takes it out again. The line
# runs boot in a login shell, with SHELL set to the user's shell, because the tmux server boot starts
# serves every tmux session opened on the machine afterwards: they get the login environment rather
# than cron's (PATH /usr/bin:/bin, /bin/sh). Boot waits up to BOOT_WAIT_SECONDS for the addresses the
# dashboard binds (a tailnet IP comes up some seconds after the network), starts the containers of
# the docker stack that could not publish their ports on such an address before it was up (docker
# does not try those again), then starts AUTOSTART, the web process and its worker, in their tmux
# sessions as the Start of their cards does, leaving one that runs already as it is. The line keeps
# this checkout, the port, conda's folder and the media ports or addresses given here as they are
# now, so run it again after changing one. Redis, Mosquitto, Nginx and the databases installed with
# brew or apt are services of their own that come back by themselves; a MediaMTX run natively is
# not (add it: make autostart AUTOSTART='flask celery mediamtx')
AUTOSTART_MARK := openmmla-dashboard-autostart
# the variables a boot has to repeat: given to `make autostart`, they are written into the line
AUTOSTART_VARS := DASHBOARD_MEDIA_PORTS DASHBOARD_MEDIA_PORT DASHBOARD_BIND ENV_NAME BOOT_WAIT_SECONDS
autostart_vars = $(strip $(foreach v,$(AUTOSTART_VARS),$(if $(filter command line environment,$(origin $(v))),$(v)='$($(v))')))
# the crontab as it is, in $$cur; stops (the crontab untouched) when crontab -l fails for another
# reason than there being none yet, rather than install a new one over it. Only its output is the
# crontab: what it says on stderr is read apart, for that test
read_crontab = err=$$(mktemp) || exit 1; cur=$$($(CRONTAB) -l 2>"$$err") || { msg=$$(cat "$$err"); rm -f "$$err"; case "$$msg" in *"no crontab"*) cur= ;; *) echo "❌ $(CRONTAB) -l failed, the crontab is left as it is: $$msg"; exit 1 ;; esac; }; rm -f "$$err"

# the line is built in the shell (the command boot runs inside the login shell's quotes, where
# \$PATH reaches that shell unexpanded), and make is named plainly: a recipe line that names the
# MAKE variable runs even under make -n, which would install the line on a dry run
autostart:
	@command -v $(CRONTAB) >/dev/null 2>&1 || { echo "❌ $(CRONTAB) not found: install cron (sudo apt install cron), or start the dashboard by hand after a reboot"; exit 1; }
	@test -n '$(CONDA_BASE)' && test -f '$(CONDA_BASE)/etc/profile.d/conda.sh' || { echo "❌ conda not found: run this where conda is on the PATH, a boot needs its folder"; exit 1; }
	@bash_bin=$$(command -v bash) || { echo "❌ bash not found"; exit 1; }; \
	login_shell=$${SHELL:-$$bash_bin}; \
	boot="cd '$(CURDIR)' && PATH='$(CONDA_BASE)/condabin':\\\$$PATH exec make boot DASHBOARD_PORT='$(DASHBOARD_PORT)' CONDA_BASE='$(CONDA_BASE)' AUTOSTART='$(AUTOSTART)'$(if $(autostart_vars),$(space)$(autostart_vars))"; \
	line="@reboot mkdir -p '$(LOG_DIR)' && SHELL='$$login_shell' '$$bash_bin' -lc \"$$boot\" >> '$(LOG_DIR)/boot.log' 2>&1 # $(AUTOSTART_MARK)"; \
	$(read_crontab); \
	{ if [ -n "$$cur" ]; then printf '%s\n' "$$cur" | grep -vF '$(AUTOSTART_MARK)'; fi; echo "$$line"; } | $(CRONTAB) - && \
	echo "✅ After a reboot cron runs: $$line"

no-autostart:
	@command -v $(CRONTAB) >/dev/null 2>&1 || { echo "No $(CRONTAB) on this machine: nothing to take out"; exit 0; }
	@$(read_crontab); \
	if ! printf '%s\n' "$$cur" | grep -qF '$(AUTOSTART_MARK)'; then echo "The crontab has no autostart line: nothing to take out"; exit 0; fi; \
	printf '%s\n' "$$cur" | grep -vF '$(AUTOSTART_MARK)' | $(CRONTAB) - && echo "✅ The dashboard no longer starts after a reboot"

# each target of AUTOSTART on its own, so one that fails (or whose session a Start opened while boot
# waited) does not keep the others from starting
boot: wait-for-bind infra-after-boot
	@for target in $(AUTOSTART); do \
		if tmux has-session -t "=$$target" 2>/dev/null; then echo "$$target runs already: left as it is"; \
		else $(MAKE) --no-print-directory $$target || echo "⚠️ $$target did not start"; fi; \
	done

# waits until each address the dashboard binds is on this machine, BOOT_WAIT_SECONDS at most, by
# binding a socket to it (127.0.1.1 is always there on Linux, a tailnet IP once tailscale is up)
BIND_PYTHON = $(or $(wildcard $(CONDA_BASE)/bin/python3),python3)
wait-for-bind:
	@echo "🔁 Boot at $$(date '+%Y-%m-%d %H:%M:%S'): waiting for $(DASHBOARD_BIND)"
	@for address in $(DASHBOARD_BIND); do \
		waited=0; \
		until $(BIND_PYTHON) -c 'import socket, sys; a = sys.argv[1]; socket.socket(socket.AF_INET6 if ":" in a else socket.AF_INET).bind((a, 0))' "$$address" 2>/dev/null; do \
			if [ $$waited -ge $(BOOT_WAIT_SECONDS) ]; then echo "⚠️ $$address is not on this machine after $(BOOT_WAIT_SECONDS)s: starting anyway"; break; fi; \
			sleep 5; waited=$$((waited + 5)); \
		done; \
	done

# the containers of the docker stack (docker/docker-compose.infra.yml) that failed to start when
# docker came up because an address they publish on (INFRA_BIND_ADDRESS, a tailnet IP) was not there
# yet: docker leaves those down, so boot starts them once the address is up. A container stopped on
# purpose carries no such error and stays stopped
INFRA_PROJECT ?= openmmla-infra
infra-after-boot:
	@command -v docker >/dev/null 2>&1 || exit 0; \
	for c in $$(docker ps -a --filter 'label=com.docker.compose.project=$(INFRA_PROJECT)' --filter status=created --filter status=exited --format '{{.Names}}' 2>/dev/null); do \
		case "$$(docker inspect -f '{{.State.Error}}' "$$c" 2>/dev/null)" in \
			*"assign requested address"*) echo "🔁 Starting $$c, which could not publish its ports before their address was up"; \
				docker start "$$c" >/dev/null || echo "⚠️ $$c did not start: docker logs $$c" ;; \
		esac; \
	done

mediamtx:
	@echo "🔁 Starting MediaMTX"
	@command -v $(MEDIAMTX_BIN) >/dev/null 2>&1 || { echo "❌ $(MEDIAMTX_BIN) not found: brew install mediamtx, or unpack a release into /usr/local/bin (docs/streaming/index.md)"; exit 1; }
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
				 $(if $(filter flask,$(FILTERED_SERVICES)),$(DASHBOARD_PORT) $(DASHBOARD_MEDIA_LISTEN),); do \
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
