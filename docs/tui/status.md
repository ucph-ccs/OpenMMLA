# Status tab

The Status tab shows what is running and where, refreshed every five seconds while the tab is open. It has nothing to do with the recording sessions of the [Sessions tab](sessions.md).

## Columns

| Column | What it shows |
|---|---|
| `Service` | the service, by its console name |
| `Host` | the host its card is on: the machine System Settings name for a system service, else the host the card was last pointed at |
| `Status` | `Running` or `Stopped`; the ASR and VFA servers show how many of their containers answer (`Running 4/6`); `? (Refresh)` for a service whose host only **Refresh** asks |
| `Port` | the port probed |
| `tmux` | the tmux session the service runs in and since when (`flask · since 09-17 13:40`): the dashboard and its worker, the MLLM Server, a native MediaMTX, streams and recorders; `-` for databases and brokers, which run under Homebrew, systemd or Docker |

The rows come from the Launcher's own picture of the deployment, so the two tabs agree: each service is listed on the host its card is on and probed the way its sidebar marker is. Bases and camera tools run in their own terminals and are not listed. Any other tmux session on this machine is, by name.

## Which services are listed

- A running service is always listed.
- A stopped service gets a row only when System Settings put it on a named machine: it is part of the deployment, so its being down is worth seeing. One whose saved address still reads `<uber-server>` is listed too, with the state of the port on its card's host; the placeholder is never looked up.
- Everything else that is stopped is counted in the summary line. **Show all** lists it, with the host of its card.

## Buttons

| Button | What it does |
|---|---|
| **Refresh** | runs the full pass over SSH, and looks on every reachable host for system services whose address is `localhost`; Windows hosts are skipped |
| **Show all** | lists the stopped services the table leaves out; it then reads **Running only**, which hides them again |
| **View Logs** | shows the selected row's logs: container, Homebrew or journald logs for a system service, the tmux pane for a session, over SSH when the row's host matches an SSH profile |

The five-second refresh only asks what can be asked from here: this machine, and addresses that name a machine. A service whose host has to be logged into shows `? (Refresh)` until **Refresh** runs.

For the ASR and VFA servers' logs, use **Logs** on their Launcher card.
