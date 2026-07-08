"""Utility script to bootstrap Google OAuth credentials.

Run this once to generate/refresh the ``token.json`` file used by the
application tools that send email or interact with Google Calendar.

Usage (from repo root):
    python google_credentials/bootstrap_oauth.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Default scopes cover Gmail send and full Calendar access
DEFAULT_SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar",
]


def _resolve_path(path_like: str, base_dir: Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    default_secret = _find_default_secret(base_dir)
    parser = argparse.ArgumentParser(description="Bootstrap Google OAuth credentials")
    parser.add_argument(
        "--secret",
        default=default_secret,
        help=(
            "Path to the OAuth client secret JSON file (defaults to client_secret.json or "
            "the first client_secret_*.json found in google_credentials/)"
        ),
    )
    parser.add_argument(
        "--token",
        default=base_dir / "token.json",
        help="Path where the authorized user token will be stored",
    )
    parser.add_argument(
        "--scope",
        dest="scopes",
        action="append",
        help="Additional OAuth scope(s) to request (can be repeated).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Use a command-line flow instead of opening a browser window",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port for the local server used during the OAuth flow (0 lets the system pick)",
    )
    args = parser.parse_args()
    args.base_dir = base_dir
    args.secret = _resolve_path(str(args.secret), base_dir)
    args.token = _resolve_path(str(args.token), base_dir)

    scopes = list(DEFAULT_SCOPES)
    if args.scopes:
        scopes.extend(args.scopes)
    args.scopes = sorted(set(scopes))
    return args


def _find_default_secret(base_dir: Path) -> Path:
    explicit = base_dir / "client_secret.json"
    if explicit.exists():
        return explicit

    matches = sorted(base_dir.glob("client_secret_*.json"))
    if matches:
        return matches[0]

    return explicit


def _load_existing_credentials(token_path: Path, scopes: list[str]) -> Credentials | None:
    if not token_path.exists():
        return None

    try:
        creds = Credentials.from_authorized_user_file(token_path, scopes)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load existing token (will request a new one): {exc}")
        return None

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            token_path.write_text(creds.to_json())
            print(f"Refreshed existing credentials -> {token_path}")
            return creds
        except Exception as exc:  # noqa: BLE001
            print(f"Could not refresh credentials, starting new flow: {exc}")

    return None


def _run_flow(secret_path: Path, scopes: list[str], use_browser: bool, port: int) -> Credentials:
    if not secret_path.exists():
        raise FileNotFoundError(
            f"OAuth client secret file not found: {secret_path}. "
            "Download it from Google Cloud Console and retry."
        )

    flow = InstalledAppFlow.from_client_secrets_file(str(secret_path), scopes=scopes)
    if use_browser:
        creds = flow.run_local_server(port=port)
    else:
        creds = flow.run_console()  # type: ignore[attr-defined]
    return cast(Credentials, creds)


def main() -> int:
    args = _parse_args()

    args.token.parent.mkdir(parents=True, exist_ok=True)

    creds = _load_existing_credentials(args.token, args.scopes)
    if creds is None:
        print("Starting OAuth flow...")
        creds = _run_flow(
            args.secret,
            scopes=args.scopes,
            use_browser=not args.no_browser,
            port=args.port,
        )
        args.token.write_text(creds.to_json())
        mask = "***" if args.token.name.lower().endswith(".json") else ""
        print(f"Saved new credentials to {args.token}{mask}")
    else:
        print(f"Existing credentials are valid -> {args.token}")

    print("Scopes granted:")
    for scope in args.scopes:
        print(f"  - {scope}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
