# src/ingkit/utils/cli.py
# argparse wrapper for ingkit scripts

import argparse
from collections.abc import Sequence
from typing import Any


class CLI:
    # TODO: add support for subcommands

    def __init__(self, description: str | None = None) -> None:
        self.parser = argparse.ArgumentParser(description=description)

    def add_argument(self, *args: Any, **kwargs: Any) -> argparse.Action:
        return self.parser.add_argument(*args, **kwargs)

    def parse_args(self, args: Sequence[str] | None = None) -> argparse.Namespace:
        return self.parser.parse_args(args)
