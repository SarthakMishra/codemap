# CodeMap Daemon Deployment

This directory contains deployment configurations and scripts for running the CodeMap daemon as a system service.

## Deployment Options

Two deployment options are provided:

1. **systemd** (recommended for Linux): Uses systemd user units for managing the daemon
2. **Supervisor**: Cross-platform process management system (Linux, macOS)

## Systemd Deployment (Linux)

The systemd configuration runs CodeMap as a user service, which means:
- It runs under your user account
- It has access to your user environment
- It doesn't require root privileges
- It will start automatically when you log in (if enabled)

### Installation

1. Make sure you are in the CodeMap project root directory
2. Run the installation script:

```bash
bash deployment/systemd/install.sh
```

3. Enable and start the service:

```bash
# Enable automatic startup at login
systemctl --user enable codemap.service

# Start the service now
systemctl --user start codemap.service
```

### Management Commands

```bash
# Check status
systemctl --user status codemap.service

# Stop the service
systemctl --user stop codemap.service

# Restart the service
systemctl --user restart codemap.service

# View logs
journalctl --user -u codemap.service

# View logs in real-time
journalctl --user -u codemap.service -f
```

## Supervisor Deployment (Linux, macOS)

Supervisor is a cross-platform process control system that can start/stop/monitor processes.

### Prerequisites

Install Supervisor:

**Ubuntu/Debian:**
```bash
sudo apt-get install supervisor
```

**macOS:**
```bash
brew install supervisor
```

### Installation

1. Make sure you are in the CodeMap project root directory
2. Run the installation script:

```bash
bash deployment/supervisor/install.sh
```

This script will:
- Create necessary log directories
- Install the configuration file in the appropriate location
- Reload the supervisor configuration

### Management Commands

If supervisor was installed system-wide (the typical case), use:

```bash
# Start the service
sudo supervisorctl start codemap

# Check status
sudo supervisorctl status codemap

# Stop the service
sudo supervisorctl stop codemap

# Restart the service
sudo supervisorctl restart codemap

# View logs
sudo tail -f ~/.codemap/logs/codemap_supervisor.out.log
```

If using a user-level supervisor installation:

```bash
# Same commands but without sudo
supervisorctl start codemap
supervisorctl status codemap
# etc.
```

## Configuration and Customization

Both deployment methods use the same CodeMap configuration system. The daemon will look for a configuration file in these locations (in order):

1. Path specified with the `--config` option
2. `~/.codemap/config.yml`
3. `./.codemap.yml` (in the current directory)

To customize the service configuration:

- **systemd**: Edit `~/.config/systemd/user/codemap.service`
- **supervisor**: Edit the configuration in the supervisor config directory (see install.sh output)

After modifying the service configuration, reload and restart the service. 