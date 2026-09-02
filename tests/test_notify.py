"""Email delivery: the gating rules, message construction and failure handling.

Every test here mocks smtplib. Nothing in this file sends mail.
"""

from __future__ import annotations

import builtins
import smtplib
import ssl

import pytest

from fi_agent.config import EmailSettings
from fi_agent.data.store import Store
from fi_agent.notify import (
    PASSWORD_ENV,
    build_message,
    load_password,
    maybe_send_report,
    new_movers,
    send,
    subject_line,
    text_summary,
    tls_context,
)
from fi_agent.schemas import AnalystResult, EvidenceItem, Finding


@pytest.fixture
def email_settings() -> EmailSettings:
    return EmailSettings(
        enabled=True,
        to=["kevin.wood75@gmail.com"],
        from_address="kevin.wood75@gmail.com",
        smtp_host="smtp.example.com",
        smtp_port=587,
    )


@pytest.fixture
def finding(mover, articles) -> Finding:
    return Finding(
        mover=mover,
        articles=articles,
        analysis=AnalystResult(
            headline="Nvidia slides with the chip sector",
            narrative="Peers fell together after a supply warning.",
            driver="sector",
            confidence="medium",
            evidence=[EvidenceItem(claim="SK Hynix warned on memory supply", source_idx=0)],
        ),
    )


class FakeSMTP:
    """Records what would have been sent."""

    instances: list[FakeSMTP] = []

    def __init__(self, host, port, timeout=None, context=None):
        self.host = host
        self.messages = []
        self.logged_in_as = None
        self.started_tls = False
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self, context=None):
        self.started_tls = True

    def login(self, user, password):
        self.logged_in_as = user

    def send_message(self, message):
        self.messages.append(message)


@pytest.fixture
def fake_smtp(monkeypatch):
    FakeSMTP.instances = []
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    monkeypatch.setenv(PASSWORD_ENV, "app-password-not-real")
    return FakeSMTP


# -- the dedupe rule --------------------------------------------------------------------


def test_new_movers_ignores_names_already_flagged():
    assert new_movers(["NVDA", "META"], {"NVDA"}) == ["META"]


def test_new_movers_treats_a_first_run_as_all_new():
    assert new_movers(["NVDA", "META"], set()) == ["NVDA", "META"]


def test_new_movers_empty_when_nothing_changed():
    assert new_movers(["NVDA"], {"NVDA", "META"}) == []


def test_store_round_trips_flagged_symbols(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        store.start_run("run1")
        store.save_flagged("run1", ["NVDA", "META"])
        store.finish_run("run1", 2, "ok", "/tmp/run1")

        store.start_run("run2")
        assert store.previously_flagged("run2") == {"NVDA", "META"}


def test_previously_flagged_is_empty_on_a_first_run(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        store.start_run("run1")
        assert store.previously_flagged("run1") == set()


def test_previously_flagged_reads_only_the_latest_completed_run(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        store.start_run("run1")
        store.save_flagged("run1", ["AAPL"])
        store.finish_run("run1", 1, "ok", "/tmp/1")
        store.start_run("run2")
        store.save_flagged("run2", ["NVDA"])
        store.finish_run("run2", 1, "ok", "/tmp/2")
        store.start_run("run3")

        assert store.previously_flagged("run3") == {"NVDA"}


# -- gating -----------------------------------------------------------------------------


def test_no_email_when_disabled(email_settings, finding, fake_smtp):
    email_settings.enabled = False
    result = maybe_send_report(email_settings, [finding], ["NVDA"], set(), "<p>x</p>")
    assert not result.sent
    assert fake_smtp.instances == []


def test_no_email_when_recipients_missing(email_settings, finding, fake_smtp):
    email_settings.to = []
    result = maybe_send_report(email_settings, [finding], ["NVDA"], set(), "<p>x</p>")
    assert not result.sent
    assert "unset" in result.detail


def test_no_email_when_nothing_flagged(email_settings, fake_smtp):
    result = maybe_send_report(email_settings, [], [], set(), "<p>x</p>")
    assert not result.sent
    assert fake_smtp.instances == []


def test_no_email_when_the_same_name_is_still_flagged(email_settings, finding, fake_smtp):
    """The whole point of the dedupe rule: a name flagged all afternoon mails once."""
    result = maybe_send_report(email_settings, [finding], ["NVDA"], {"NVDA"}, "<p>x</p>")
    assert not result.sent
    assert "newly flagged" in result.detail
    assert fake_smtp.instances == []


def test_email_sent_for_a_newly_flagged_name(email_settings, finding, fake_smtp):
    result = maybe_send_report(email_settings, [finding], ["NVDA"], {"AAPL"}, "<p>body</p>")
    assert result.sent
    assert len(fake_smtp.instances) == 1
    assert len(fake_smtp.instances[0].messages) == 1


def test_email_uses_tls_and_logs_in(email_settings, finding, fake_smtp):
    maybe_send_report(email_settings, [finding], ["NVDA"], set(), "<p>body</p>")
    server = fake_smtp.instances[0]
    assert server.started_tls
    assert server.logged_in_as == "kevin.wood75@gmail.com"


def test_force_sends_even_when_nothing_is_new(email_settings, finding, fake_smtp):
    """--force-email is 'send me the current picture', so the dedupe must not block it."""
    result = maybe_send_report(
        email_settings, [finding], ["NVDA"], {"NVDA"}, "<p>x</p>", force=True
    )
    assert result.sent
    assert len(fake_smtp.instances[0].messages) == 1


def test_force_cannot_send_an_empty_report(email_settings, fake_smtp):
    """Forcing must not invent something to report when nothing flagged."""
    result = maybe_send_report(email_settings, [], [], set(), "<p>x</p>", force=True)
    assert not result.sent
    assert "nothing to send" in result.detail
    assert fake_smtp.instances == []


def test_force_does_not_override_the_disabled_switch(email_settings, finding, fake_smtp):
    email_settings.enabled = False
    result = maybe_send_report(
        email_settings, [finding], ["NVDA"], set(), "<p>x</p>", force=True
    )
    assert not result.sent
    assert fake_smtp.instances == []


# -- message shape ----------------------------------------------------------------------


def test_subject_names_the_movers_and_their_moves(finding):
    assert subject_line([finding], ["NVDA"]) == "[fi-agent] NVDA -4.6%"


def test_subject_truncates_a_long_list(finding):
    subject = subject_line([finding], ["NVDA", "A", "B", "C", "D"])
    assert subject.endswith("+2 more")


def test_message_carries_html_and_a_plain_text_alternative(email_settings, finding):
    message = build_message(
        email_settings, "subject", "<p>the report</p>", text_summary([finding], ["NVDA"])
    )
    assert message.is_multipart()
    types = {part.get_content_type() for part in message.walk()}
    assert "text/html" in types and "text/plain" in types
    assert message["To"] == "kevin.wood75@gmail.com"


def test_report_is_only_attached_when_asked(email_settings, finding, tmp_path):
    report = tmp_path / "report.html"
    report.write_text("<html>full report</html>")

    without = build_message(email_settings, "s", "<p>x</p>", "x", None)
    assert not [p for p in without.walk() if p.get_filename()]

    with_file = build_message(email_settings, "s", "<p>x</p>", "x", report)
    assert [p.get_filename() for p in with_file.walk() if p.get_filename()] == ["report.html"]


def test_text_summary_includes_the_cause_and_its_source(finding):
    text = text_summary([finding], ["NVDA"])
    assert "NVDA (NVIDIA) -4.57%" in text
    assert "Nvidia slides with the chip sector" in text
    assert "https://example.com/a" in text


# -- failure handling -------------------------------------------------------------------


def test_missing_password_is_reported_not_raised(email_settings, monkeypatch):
    monkeypatch.delenv(PASSWORD_ENV, raising=False)
    monkeypatch.setattr("fi_agent.notify.load_dotenv", lambda *a, **k: None)
    message = build_message(email_settings, "s", "<p>x</p>", "x")
    result = send(email_settings, message)
    assert not result.sent
    assert PASSWORD_ENV in result.detail


def test_bad_credentials_give_an_actionable_message(email_settings, monkeypatch):
    monkeypatch.setenv(PASSWORD_ENV, "wrong")

    class Rejecting(FakeSMTP):
        def login(self, user, password):
            raise smtplib.SMTPAuthenticationError(535, b"bad")

    monkeypatch.setattr(smtplib, "SMTP", Rejecting)
    result = send(email_settings, build_message(email_settings, "s", "<p>x</p>", "x"))
    assert not result.sent
    assert "App Password" in result.detail


def test_credential_never_appears_in_the_failure_detail(email_settings, monkeypatch):
    secret = "sup3rs3cret-app-password"
    monkeypatch.setenv(PASSWORD_ENV, secret)

    class Rejecting(FakeSMTP):
        def login(self, user, password):
            raise smtplib.SMTPAuthenticationError(535, secret.encode())

    monkeypatch.setattr(smtplib, "SMTP", Rejecting)
    result = send(email_settings, build_message(email_settings, "s", "<p>x</p>", "x"))
    assert secret not in result.detail


def test_unreachable_server_does_not_raise(email_settings, monkeypatch):
    monkeypatch.setenv(PASSWORD_ENV, "x")

    def boom(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(smtplib, "SMTP", boom)
    result = send(email_settings, build_message(email_settings, "s", "<p>x</p>", "x"))
    assert not result.sent
    assert "connection refused" in result.detail


def test_send_failure_does_not_break_the_run(email_settings, finding, monkeypatch):
    """A dead mail server must not cost you the report already written to disk."""
    monkeypatch.setenv(PASSWORD_ENV, "x")
    monkeypatch.setattr(
        smtplib, "SMTP", lambda *a, **k: (_ for _ in ()).throw(OSError("down"))
    )
    result = maybe_send_report(email_settings, [finding], ["NVDA"], set(), "<p>x</p>")
    assert not result.sent


# -- TLS trust store --------------------------------------------------------------------


def test_tls_context_has_certificate_authorities():
    """Regression. Python from python.org ships an empty trust store, so
    ssl.create_default_context() loads zero CAs and every send failed with
    CERTIFICATE_VERIFY_FAILED."""
    assert len(tls_context().get_ca_certs()) > 0


def test_tls_context_uses_the_certifi_bundle():
    import certifi

    expected = len(ssl.create_default_context(cafile=certifi.where()).get_ca_certs())
    assert len(tls_context().get_ca_certs()) == expected


def test_tls_context_falls_back_when_certifi_is_missing(monkeypatch):
    """A missing certifi must degrade to the platform store, not crash the send."""
    real_import = builtins.__import__

    def no_certifi(name, *args, **kwargs):
        if name == "certifi":
            raise ImportError("no certifi")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_certifi)
    assert isinstance(tls_context(), ssl.SSLContext)


def test_password_read_from_environment(monkeypatch):
    monkeypatch.setenv(PASSWORD_ENV, "  spaced  ")
    assert load_password() == "spaced"


def test_blank_password_reads_as_absent(monkeypatch):
    monkeypatch.setenv(PASSWORD_ENV, "   ")
    monkeypatch.setattr("fi_agent.notify.load_dotenv", lambda *a, **k: None)
    assert load_password() is None
