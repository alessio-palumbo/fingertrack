import argparse
import json
from typing import Any, Protocol

import requests


class FingersConsumer(Protocol):
    """Defines an interface for consuming finger state outputs."""

    def consume(self, fingers: Any) -> None: ...


class StdoutConsumer:
    """Default consumer that prints finger states as JSON to stdout."""

    def consume(self, fingers: Any) -> None:
        print(json.dumps(fingers), flush=True)


class HttpConsumer:
    def __init__(self, url: str):
        self.url = url

    def consume(self, fingers: Any) -> None:
        requests.post(self.url, json=fingers)


def get_consumer_from_args() -> FingersConsumer:
    """Returns the consumer as defined in the args or the default"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--consumer",
        choices=["stdout", "http"],
        default="stdout",
        help="Select output consumer (default: stdout)",
    )
    parser.add_argument("--url", help="URL for http consumer")
    args = parser.parse_args()

    if args.consumer == "http":
        return HttpConsumer(args.url)
    return StdoutConsumer()
