#!/usr/bin/env bash
# Runs a command and runs it again when it exits with an error, as `restart: on-failure` does for the
# containers of the Docker stacks. The Makefile starts the dashboard's web process and its worker
# through it in their tmux panes:
#
#   bash restart_on_failure.sh gunicorn -k gevent -w 1 -b 127.0.0.1:5050 dashboard:app
#
# A run that exits 0 ends it, and so does Ctrl+C, which the Stop of the cards (make stop-flask,
# stop-celery) sends: it reaches this script too, as the pane's foreground, and the command it
# stopped is not started again whatever it exits with (the Celery worker exits 1 after its warm
# shutdown). Any other failure starts the command again after
# RESTART_DELAY seconds, at most RESTART_LIMIT times in a row; a run that lasted QUICK_SECONDS or
# longer starts the count again, so a dashboard that crashes once a week keeps coming back while one
# that cannot start (its port taken, a package missing) says why in its pane and stops there.

RESTART_LIMIT=${RESTART_LIMIT:-3}
RESTART_DELAY=${RESTART_DELAY:-5}
QUICK_SECONDS=${QUICK_SECONDS:-60}

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <command> [<argument>...]" >&2
    exit 2
fi

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

# bash runs the trap once the command (or the sleep) it waits for has ended; the command itself
# keeps the default Ctrl+C behaviour
interrupted=0
trap 'interrupted=1' INT

restarts=0
while true; do
    echo "[$(stamp)] restart_on_failure: starting $1"
    started=$SECONDS
    "$@"
    status=$?
    if [ "$status" -eq 0 ] || [ "$interrupted" -eq 1 ]; then
        exit "$status"
    fi
    # a run that lasted a while failed for a new reason: it does not count against the ones before
    if [ $((SECONDS - started)) -ge "$QUICK_SECONDS" ]; then
        restarts=0
    fi
    if [ "$restarts" -ge "$RESTART_LIMIT" ]; then
        echo "[$(stamp)] restart_on_failure: $1 exited with $status after $restarts restarts in a row: giving up" >&2
        exit "$status"
    fi
    restarts=$((restarts + 1))
    echo "[$(stamp)] restart_on_failure: $1 exited with $status: restart $restarts of $RESTART_LIMIT in ${RESTART_DELAY}s" >&2
    sleep "$RESTART_DELAY"
    if [ "$interrupted" -eq 1 ]; then
        exit "$status"
    fi
done
