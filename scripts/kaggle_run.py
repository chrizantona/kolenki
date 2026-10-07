"""Launch/read Kaggle jobs; authentication stays in the environment or Kaggle's own credential store."""
import argparse
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi


def serial(value):
    if hasattr(value, "to_dict"):
        result = serial(value.to_dict())
        if isinstance(result, dict) and hasattr(value, "status"):
            status = value.status
            result["status"] = status.name if hasattr(status, "name") else str(status)
        return result
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["quota", "push", "status", "logs"])
    p.add_argument("--kernel")
    p.add_argument("--folder", type=Path)
    p.add_argument("--timeout", type=int, default=12600)
    p.add_argument("--record", type=Path)
    args = p.parse_args()
    if args.action == "push" and args.folder is None:
        p.error("push requires --folder with a reviewed notebook and kernel-metadata.json")
    if args.action in ("status", "logs") and not args.kernel:
        p.error("status/logs require --kernel owner/slug")
    api = KaggleApi()
    api.authenticate()
    if args.action == "quota":
        result = api.quota_view()
    elif args.action == "push":
        result = api.kernels_push(str(args.folder), timeout=str(args.timeout), acc="NvidiaTeslaT4")
    elif args.action == "status":
        result = api.kernels_status(args.kernel)
    else:
        result = api.kernels_logs(args.kernel)
    payload = serial(result)
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if isinstance(payload, dict) and payload.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
