from __future__ import annotations

import time

from owl._monitor.ring import MonitorRing
from owl._monitor.server import start_monitor_server, stop_monitor_server
from owl._monitor.snapshot import MonitorSnapshot


def main() -> None:
    ring = MonitorRing()
    handle = start_monitor_server(ring)

    print(f"monitor server started: {handle.address}")
    print("copy this address and run:")
    print(f"python test/monitor_client.py {handle.address}")

    step = 0

    try:
        while True:
            step += 1

            ring.append(
                MonitorSnapshot.from_train_step(
                    epoch=0,
                    step=step,
                    model_metrics={
                        "router_weight": round(0.5 + step * 0.01, 4),
                    },
                    loss_metrics={
                        "bce_loss": round(1.0 / (step + 1), 4),
                        "edge_loss": round(0.5 / (step + 1), 4),
                    },
                )
            )

            print(f"append snapshot step={step}")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("stopping monitor server...")
    finally:
        stop_monitor_server(handle)


if __name__ == "__main__":
    main()