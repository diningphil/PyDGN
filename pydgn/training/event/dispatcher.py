from pydgn.training.event.state import State


class EventDispatcher:
    """
    Class implementing the publisher/subscribe pattern. It is used to register subscribers that implement the
    :class:`~training.event.handler.EventHandler` interface """

    def __init__(self):
        self._event_handlers = []

    def _dispatch(self, event_name: str, state: State):
        """
        Triggers the callback ``event_name`` for all subscribers (**note: order matters!**)

        Args:
            event_name (str): the name of the callback to trigger
            state (:class:`~training.event.state.State`): object holding training information
        """
        for event_handler in self._event_handlers:
            try:
                callback = getattr(event_handler, event_name)
            except AttributeError:
                # This happens when the callback does not implement a particular method
                # e.g., when using a particular engine with new events and event handlers
                callback = None

            if callback is not None:
                callback(state)

    def register(self, event_handler):
        """
        Registers a subscriber

        Args:
            event_handler: an object implementing the :class:`~training.event.handler.EventHandler` interface
        """
        self._event_handlers.append(event_handler)

    def unregister(self, event_handler):
        """
        De-registers a subscriber

        Args:
            event_handler: an object implementing the :class:`~training.event.handler.EventHandler` interface
        """
        self._event_handlers.remove(event_handler)
