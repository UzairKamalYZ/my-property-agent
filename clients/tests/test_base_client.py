import unittest

from clients.base import BaseClient


class TestBaseClient(unittest.TestCase):

    # ------------------------------------------------------------------
    # Instantiation contract
    # ------------------------------------------------------------------

    def test_should_raise_type_error_when_start_is_not_implemented(self):
        """BaseClient cannot be instantiated without a concrete start()."""
        with self.assertRaises(TypeError):
            BaseClient()  # type: ignore[abstract]

    def test_should_allow_instantiation_when_start_is_implemented(self):
        """A subclass that implements start() can be instantiated."""
        class ConcreteClient(BaseClient):
            def start(self) -> None:
                pass

        try:
            ConcreteClient()
        except TypeError as exc:
            self.fail(f"Instantiation raised unexpectedly: {exc}")

    # ------------------------------------------------------------------
    # stop() default
    # ------------------------------------------------------------------

    def test_should_not_raise_when_default_stop_is_called(self):
        """The inherited stop() is a safe no-op when not overridden."""
        class ConcreteClient(BaseClient):
            def start(self) -> None:
                pass

        try:
            ConcreteClient().stop()
        except Exception as exc:
            self.fail(f"stop() raised unexpectedly: {exc}")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def test_should_return_self_when_entering_context_manager(self):
        """__enter__ returns the client instance itself."""
        class ConcreteClient(BaseClient):
            def start(self) -> None:
                pass

        client = ConcreteClient()
        with client as ctx:
            self.assertIs(ctx, client)

    def test_should_call_stop_when_context_manager_exits_normally(self):
        """__exit__ calls stop() after the with-block completes."""
        class ConcreteClient(BaseClient):
            def __init__(self):
                self.stop_called = False

            def start(self) -> None:
                pass

            def stop(self) -> None:
                self.stop_called = True

        client = ConcreteClient()
        with client:
            pass
        self.assertTrue(client.stop_called)

    def test_should_call_stop_when_context_manager_exits_with_exception(self):
        """__exit__ calls stop() even when the with-block raises."""
        class ConcreteClient(BaseClient):
            def __init__(self):
                self.stop_called = False

            def start(self) -> None:
                pass

            def stop(self) -> None:
                self.stop_called = True

        client = ConcreteClient()
        try:
            with client:
                raise ValueError("deliberate test error")
        except ValueError:
            pass
        self.assertTrue(client.stop_called)


if __name__ == "__main__":
    unittest.main()
