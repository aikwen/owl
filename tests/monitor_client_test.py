from __future__ import annotations

import argparse

from owl._monitor.client import (
    health_from_address,
    stream_from_address,
    window_from_address,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("address", help='monitor server address, e.g. "127.0.0.1:39125"')
    args = parser.parse_args()

    print("health:")
    print(health_from_address(args.address))

    print("\nwindow:")
    snapshots = window_from_address(args.address)

    for snapshot in snapshots:
        print(snapshot)

    last_seq = snapshots[-1]["seq"] if snapshots else 0

    print(f"\nstream from seq={last_seq}:")
    for snapshot in stream_from_address(args.address, last_seq=last_seq):
        print(snapshot)


if __name__ == "__main__":
    main()