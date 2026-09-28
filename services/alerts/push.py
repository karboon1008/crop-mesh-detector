"""Push notifications to the farmer's iPhone through Apple Push Notification
service (APNs), with token-based (.p8 key) auth over HTTP/2.

Without APNs credentials in the config the service runs in dry-run mode:
every notification is still built and written to the push_log table (and
shown at GET /api/admin/push-log), just not sent. That is enough to demo
and test the whole flow without an Apple Developer account.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

SENT = "sent"
DRY_RUN = "dry_run"
INVALID_TOKEN = "invalid_token"  # the device should be forgotten
FAILED = "failed"


class DryRunSender:
    def send(self, token: str, payload: dict, collapse_id: str | None = None) -> str:
        return DRY_RUN


@dataclass
class APNsConfig:
    team_id: str
    key_id: str
    key_path: str  # the AuthKey_<key_id>.p8 downloaded from the Apple Developer portal
    bundle_id: str  # the iOS app's bundle id, sent as apns-topic
    use_sandbox: bool = True  # Xcode / TestFlight debug builds use the sandbox gateway


class APNsSender:
    PRODUCTION = "https://api.push.apple.com"
    SANDBOX = "https://api.sandbox.push.apple.com"
    TOKEN_LIFETIME_S = 50 * 60  # Apple rejects provider tokens older than 60 min

    def __init__(self, config: APNsConfig, client=None):
        import httpx

        self.config = config
        self.base_url = self.SANDBOX if config.use_sandbox else self.PRODUCTION
        self.client = client or httpx.Client(http2=True, timeout=10.0)
        self._signing_key = Path(config.key_path).read_text()
        self._token: str | None = None
        self._token_issued = 0.0

    def _provider_token(self) -> str:
        import jwt

        now = time.time()
        if self._token is None or now - self._token_issued > self.TOKEN_LIFETIME_S:
            self._token = jwt.encode(
                {"iss": self.config.team_id, "iat": int(now)}, self._signing_key,
                algorithm="ES256", headers={"kid": self.config.key_id},
            )
            self._token_issued = now
        return self._token

    def send(self, token: str, payload: dict, collapse_id: str | None = None) -> str:
        headers = {
            "authorization": f"bearer {self._provider_token()}",
            "apns-topic": self.config.bundle_id,
            "apns-push-type": "alert",
            "apns-priority": "10",
        }
        if collapse_id:
            headers["apns-collapse-id"] = collapse_id  # a newer notice replaces the older one on the lock screen
        try:
            response = self.client.post(f"{self.base_url}/3/device/{token}", json=payload, headers=headers)
        except Exception:
            return FAILED
        if response.status_code == 200:
            return SENT
        if response.status_code == 410 or (
            response.status_code == 400 and "BadDeviceToken" in response.text
        ):
            return INVALID_TOKEN
        return FAILED


def build_sender(apns: dict | None):
    if not apns:
        return DryRunSender()
    return APNsSender(APNsConfig(**apns))
