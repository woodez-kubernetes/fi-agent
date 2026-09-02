"""Email delivery of the report.

Named `notify` rather than `email` so it does not shadow the standard library module it
depends on.

The SMTP password never appears in configuration or in this file. It is read from the
`FI_AGENT_SMTP_PASSWORD` environment variable, which in practice comes from a gitignored
`.env`, so the credential cannot be committed by accident and is never logged.

Delivery failures are reported but never raised: a mail server being down must not cost
you the report that was already written to disk.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv

from fi_agent.config import EmailSettings, project_root
from fi_agent.schemas import Finding

log = logging.getLogger(__name__)

PASSWORD_ENV = "FI_AGENT_SMTP_PASSWORD"


@dataclass
class SendResult:
    sent: bool
    detail: str


def tls_context() -> ssl.SSLContext:
    """An SSL context with a usable certificate authority bundle.

    Python installed from python.org does not consult the macOS keychain and ships with
    an empty trust store - `ssl.create_default_context()` there loads zero CA
    certificates, so every TLS verification fails with CERTIFICATE_VERIFY_FAILED. The
    conventional fix is to run the installer's "Install Certificates.command", but that
    is a manual step outside the project that a Python reinstall undoes.

    Using certifi's bundle instead keeps the fix inside the codebase, where it travels
    with the project and cannot be forgotten.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except (ImportError, OSError) as exc:
        # Better to try the platform default than to refuse to send at all: on a system
        # with a properly populated store it works fine.
        log.warning("certifi unavailable (%s), falling back to the system CA store", exc)
        return ssl.create_default_context()


def load_password() -> str | None:
    """Read the SMTP password from the environment, loading .env first if present."""
    load_dotenv(project_root() / ".env", override=False)
    password = os.environ.get(PASSWORD_ENV, "").strip()
    return password or None


def new_movers(current: list[str], previous: set[str]) -> list[str]:
    """Symbols flagged now that were not flagged in the previous run."""
    return [symbol for symbol in current if symbol not in previous]


def subject_line(findings: list[Finding], symbols: list[str]) -> str:
    """A subject that is readable from a phone lock screen without opening anything."""
    by_symbol = {f.mover.symbol: f for f in findings}
    parts = []
    for symbol in symbols[:3]:
        finding = by_symbol.get(symbol)
        if finding:
            parts.append(f"{symbol} {finding.mover.quote.pct_change:+.1f}%")
        else:
            parts.append(symbol)
    if len(symbols) > 3:
        parts.append(f"+{len(symbols) - 3} more")
    return f"[fi-agent] {', '.join(parts)}"


def build_message(
    settings: EmailSettings,
    subject: str,
    html: str,
    text_fallback: str,
    attachment: Path | None = None,
) -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings.from_address
    message["To"] = ", ".join(settings.to)

    # Plain-text part first, then HTML: clients pick the richest they can render.
    message.set_content(text_fallback)
    message.add_alternative(html, subtype="html")

    if attachment is not None and attachment.exists():
        message.add_attachment(
            attachment.read_bytes(),
            maintype="text",
            subtype="html",
            filename=attachment.name,
        )
    return message


def send(settings: EmailSettings, message: EmailMessage) -> SendResult:
    """Deliver one message over SMTP. Never raises."""
    password = load_password()
    if password is None:
        return SendResult(False, f"{PASSWORD_ENV} is not set; see deploy-run.md")

    try:
        if settings.use_tls:
            with smtplib.SMTP(
                settings.smtp_host, settings.smtp_port, timeout=settings.timeout_s
            ) as server:
                server.starttls(context=tls_context())
                server.login(settings.from_address, password)
                server.send_message(message)
        else:
            with smtplib.SMTP_SSL(
                settings.smtp_host,
                settings.smtp_port,
                timeout=settings.timeout_s,
                context=tls_context(),
            ) as server:
                server.login(settings.from_address, password)
                server.send_message(message)
    except smtplib.SMTPAuthenticationError:
        # Deliberately does not echo the credential or the server's verbose reply.
        return SendResult(
            False,
            "SMTP rejected the login. For Gmail this must be a 16-character App "
            "Password, not your account password.",
        )
    except (OSError, smtplib.SMTPException) as exc:
        return SendResult(False, f"{type(exc).__name__}: {exc}")

    return SendResult(True, f"sent to {', '.join(settings.to)}")


def text_summary(findings: list[Finding], symbols: list[str]) -> str:
    """Plain-text version, for clients that will not render HTML."""
    by_symbol = {f.mover.symbol: f for f in findings}
    lines = ["Newly flagged on your watchlist:", ""]
    for symbol in symbols:
        finding = by_symbol.get(symbol)
        if finding is None:
            continue
        quote = finding.mover.quote
        lines.append(f"{symbol} ({finding.mover.name}) {quote.pct_change:+.2f}%")
        if finding.analysis:
            lines.append(f"  {finding.analysis.headline}")
            lines.append(
                f"  driver: {finding.analysis.driver.replace('_', ' ')}, "
                f"confidence: {finding.analysis.confidence}"
            )
        for claim, article in finding.cited_articles:
            lines.append(f"  - {claim}")
            lines.append(f"    {article.url}")
        lines.append("")
    lines.append("Full report attached or on disk under reports/.")
    return "\n".join(lines)


def maybe_send_report(
    settings: EmailSettings,
    findings: list[Finding],
    flagged: list[str],
    previously_flagged: set[str],
    html: str,
    report_path: Path | None = None,
    force: bool = False,
) -> SendResult:
    """Send the report if, and only if, something has newly flagged.

    `force` overrides the newly-flagged rule and sends whatever the run found, for
    "email me the current picture" on demand. It does not override the enabled flag or
    missing configuration, and it cannot invent something to send: a run with no movers
    at all still has nothing to report.

    Returns a result describing what happened, including the reasons for not sending, so
    the caller can surface it rather than failing silently.
    """
    if not settings.enabled:
        return SendResult(False, "email disabled")
    if not settings.is_configured:
        return SendResult(False, "email enabled but 'to' or 'from_address' is unset")

    if force:
        if not flagged:
            return SendResult(False, "nothing flagged, so there is nothing to send")
        fresh = list(flagged)
        log.info("forcing email for all flagged tickers: %s", ", ".join(fresh))
    else:
        fresh = new_movers(flagged, previously_flagged)

    if not fresh:
        if flagged:
            return SendResult(False, "no newly flagged tickers since the last run")
        return SendResult(False, "nothing flagged")

    log.info("emailing report for newly flagged: %s", ", ".join(fresh))
    message = build_message(
        settings,
        subject_line(findings, fresh),
        html,
        text_summary(findings, fresh),
        report_path if settings.attach_report else None,
    )
    result = send(settings, message)
    if not result.sent:
        log.warning("email not sent: %s", result.detail)
    return result
