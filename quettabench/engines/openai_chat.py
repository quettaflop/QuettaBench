"""Import path kept for existing callers; the body lives in drivers.HttpDriver."""

from quettabench.engines.drivers import HttpDriver

send_request = HttpDriver("chat", "openai").send_request
