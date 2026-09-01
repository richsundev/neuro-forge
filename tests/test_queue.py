"""Unit test for the worker queue's timeout handling — regression test for a real bug found while
verifying `docker compose up`: the client-side socket timeout raced the server-side BLPOP timeout
and crashed the worker process on every empty poll instead of returning None."""

from __future__ import annotations

from unittest.mock import MagicMock

import redis

from neuroforge.experiments.queue import dequeue_experiment, enqueue_experiment


def test_dequeue_returns_none_on_client_socket_timeout():
    client = MagicMock()
    client.blpop.side_effect = redis.exceptions.TimeoutError("Timeout reading from socket")
    assert dequeue_experiment(timeout=5, client=client) is None


def test_dequeue_returns_experiment_id_when_present():
    client = MagicMock()
    client.blpop.return_value = ("neuroforge:experiments:queue", "exp-42")
    assert dequeue_experiment(timeout=5, client=client) == "exp-42"


def test_dequeue_returns_none_when_queue_empty():
    client = MagicMock()
    client.blpop.return_value = None
    assert dequeue_experiment(timeout=5, client=client) is None


def test_enqueue_pushes_to_the_queue_key():
    client = MagicMock()
    enqueue_experiment("exp-1", client=client)
    client.rpush.assert_called_once_with("neuroforge:experiments:queue", "exp-1")
