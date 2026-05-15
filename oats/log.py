"""Logging utilities for oats."""
#!/usr/bin/env python3

import os
from datetime import datetime, timezone


CORE_LOGGER = None

class Lg:
    """
    A very basic logger that prints to stdout.  Designed for easy
    monkey-patching and debugging.

    disable colors with ``export COLORS_ENABLED=0`` colors are enabled by default.
    """

    def __init__(self, name: str = "lg", colors_enabled: bool | None = None, file: str | None = None) -> None:
        """Initialize the logger with the given name and options."""
        from rich.console import Console
        self.console = Console()
        self.file = file
        self.name = name
        self.colors_enabled = colors_enabled
        self.show_logs = False
        self.logs = []
        if self.colors_enabled is None:
            self.colors_enabled = os.getenv('COLORS_ENABLED', '1') == '1'

    def log(self, level, m: str) -> None:
        """Logs a message with a given level."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        use_level = level.lower().strip().lstrip()
        md_test = m.lstrip()
        self.logs.append(md_test)
        # if the log line looks like it could be markdown table of contents header tag then handle it with the rich.Markdown
        if md_test[0] == '#' and len(md_test) > 4 and md_test[1] in ['#', ' ']:
            if md_test[1] == ' ':
                md_test = '##' + md_test[1:]
            from rich.markdown import Markdown
            use_str = Markdown(md_test)
            if use_level in ['i', 'inf', 'info']:
                self.console.print(use_str)
            elif use_level in ['d', 'debug']:
                self.console.print(f"[gray] - DEBUG - {timestamp} - DEBUG_START[/gray]")
                self.console.print(use_str)
                self.console.print(f"[gray] - DEBUG - {timestamp} - DEBUG_END[/gray]")
            elif use_level in ['w', 'warn', 'warning']:
                self.console.print(f"[orange] - WARN - {timestamp} - WARN_START[/orange]")
                self.console.print(use_str)
                self.console.print(f"[orange] - WARN - {timestamp} - WARN_END[/orange]")
            elif use_level in ['e', 'err', 'error']:
                self.console.print(f"[red] - ERROR - {timestamp} - ERROR_START[/red]")
                self.console.print(use_str)
                self.console.print(f"[red] - ERROR - {timestamp} - ERROR_END[/red]")
            elif use_level in ['c', 'crit', 'critical']:
                self.console.print(f"[red] - ERROR - {timestamp} - CRIT_START[/red]")
                self.console.print(use_str)
                self.console.print(f"[red] - ERROR - {timestamp} - CRIT_END[/red]")
            elif use_level in ['f', 'fail', 'failed', 'failure']:
                self.console.print(f"[red] - FAIL - {timestamp} - FAIL_START[/red]")
                self.console.print(use_str)
                self.console.print(f"[red] - FAIL - {timestamp} - FAIL_END[/red]")
            elif use_level in ['g', 'green', 'good']:
                self.console.print(f"[green] GOOD - {timestamp} - GOOD_START[/green]")
                self.console.print(use_str)
                self.console.print(f"[green] GOOD - {timestamp} - GOOD_END[/green]")
            elif use_level in ['p', 'pass', 'pass']:
                self.console.print(f"[green] PASSED - {timestamp} - PASS_START[/green]")
                self.console.print(use_str)
                self.console.print(f"[green] PASSED - {timestamp} - PASS_END[/green]")
            elif use_level in ['s', 'startup']:
                self.console.print(f"[green] STARTUP - {timestamp} - STARTUP_START[/green]")
                self.console.print(use_str)
                self.console.print(f"[green] STARTUP - {timestamp} - STARTUP_END[/green]")
            else:
                self.console.print(use_str)
        else:
            if use_level == 'info':
                print(f"{timestamp} - {self.name} - {level} - {m}", flush=True)
            elif use_level == 'warning':
                if self.colors_enabled:
                    print(f"\033[33m{timestamp} - {self.name} - {level} - {m}\033[0m", flush=True)
                else:
                    print(f"{timestamp} - {self.name} - {level} - {m}", flush=True)
            elif use_level == 'error':
                if self.colors_enabled:
                    print(f"\033[31m{timestamp} - {self.name} - {level} - {m}\033[0m", flush=True)
                else:
                    print(f"{timestamp} - {self.name} - {level} - {m}", flush=True)
            elif use_level == 'critical':
                if self.colors_enabled:
                    print(f"\033[31m{timestamp} - {self.name} - {level} - {m}\033[0m", flush=True)
                else:
                    print(f"{timestamp} - {self.name} - {level} - {m}", flush=True)
            elif use_level == 'debug':
                if self.colors_enabled:
                    print(f"\033[1;37m{timestamp} - {self.name} - {level} - {m}\033[0m", flush=True)
                else:
                    print(f"{timestamp} - {self.name} - {level} - {m}", flush=True)
            else:
                print(f"{timestamp} - {self.name} - {level} - {m}", flush=True)

    def debug(self, m: str) -> None:
        """Log a debug-level message."""
        self.log("DEBUG", m)

    def info(self, m: str) -> None:
        """Log an info-level message."""
        self.log("INFO", m)

    def warn(self, m: str) -> None:
        """Log a warning-level message."""
        self.log("WARN", m)

    def warning(self, m: str) -> None:
        """Log a warning-level message."""
        self.log("WARNING", m)

    def err(self, m: str) -> None:
        """Log an error-level message."""
        self.log("ERROR", m)

    def error(self, m: str) -> None:
        """Log an error-level message."""
        self.log("ERROR", m)

    def critical(self, m: str) -> None:
        """Log a critical-level message."""
        self.log("CRITICAL", m)

    def good(self, m: str) -> None:
        """Log a success/good message."""
        self.log("g", m)

    def p(self, m: str) -> None:
        """Log a pass message."""
        self.log("p", m)

    def fail(self, m: str) -> None:
        """Log a failure message."""
        self.log("f", m)

    def startup(self, m: str) -> None:
        """Log a startup message."""
        self.log("s", m)

    def agent_log(self, m: str) -> None:
        """Log an agent-specific message."""
        self.log("AGENTLOG", m)

    def get_file(self) -> str | None:
        """Return the log file path, if set."""
        return self.file

    def save_all(self, file: str | None) -> None:
        """Save all accumulated logs to the configured file."""
        if file is None and self.file is not None:
            file = self.file
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if len(self.logs) > 0:
            str_logs = '\n'.join(self.logs)
            # append to the existing log
            with open(file, 'a') as f:
                f.write(str_logs)

    def save(self, file: str | None, num_logs: int = 10, total_len: int = 500, is_append: bool = False) -> None:
        """Write the last *num_logs* entries to *file* (truncated to *total_len* chars)."""
        if file is None and self.file is not None:
            file = self.file
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if len(self.logs) > 0:
            str_logs = '\n'.join(self.logs[-num_logs:])
            if len(str_logs) > 0:
                if is_append:
                    # append to the existing log
                    with open(file, 'a') as f:
                        f.write(str_logs[-total_len:])
                else:
                    with open(file, 'w') as f:
                        f.write(str_logs[-total_len:])

    def get_logs(self) -> list[str]:
        """Return the list of accumulated log strings."""
        return self.logs

    def set_file(self, file: str) -> None:
        """Set the log file path."""
        self.file = file

    def enable_logs(self) -> None:
        """Enable log output to console."""
        self.show_logs = True

def create_log(n: str = "lg", colors: bool = True, file: str | None = None):
    """
    Creates a logger Lg instance.

    Args:
        name: The name of the logger (for identification in logs).
        colors: enable colors in the log
        file: enable saving the logs to this file

    Returns:
        A Lg instance.
    """
    return gl(n=n, colors=colors, file=file)

def cl(n: str = "lg", colors: bool = True, file: str | None = None):
    """
    Creates a logger Lg instance.

    Args:
        name: The name of the logger (for identification in logs).
        colors: enable colors in the log
        file: enable saving the logs to this file

    Returns:
        A Lg instance.
    """
    return gl(n=n, colors=colors, file=file)

def gl(n: str | None = None, colors: bool | None = None, file: str | None = None):
    """
    Initializes a global logger Lg instance - disabled by default

    Args:
        name: The name of the logger (for identification in logs).
        colors: enable colors in the log
        file: enable saving the logs to this file

    Returns:
        A Lg instance.
    """
    if False:
        global CORE_LOGGER
        if CORE_LOGGER is None:
            if n is None:
                n = os.getenv('CORE_LOGGER_NAME', 'lg')
            CORE_LOGGER = Lg(name=n, colors_enabled=colors, file=file)
        CORE_LOGGER.name = n
        return CORE_LOGGER
    else:
        CORE_LOGGER = Lg(name=n, colors_enabled=colors, file=file)
        return CORE_LOGGER

# Example Usage
if __name__ == '__main__':
    # Standard Usage
    log: Lg = create_log(n="main")
    log.info("application started")
    log.debug("debugging information")
    log.error("an error occurred!")
