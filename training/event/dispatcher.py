class EventDispatcher:
    """ Simple class implementing the publisher/subscribe pattern. Callbacks objects should implement the
    training.core.event.handler.EventHandler interface """

    def __init__(self):
        self._event_handlers = []

    def _dispatch(self, event_name, state):
        for event_handler in self._event_handlers:
            callback = getattr(event_handler, event_name)
            callback(state)

    def register(self, event_handler):
        self._event_handlers.append(event_handler)

    def unregister(self, event_handler):
        self._event_handlers.remove(event_handler)

    def register_all(self):
        for event_handler in self._event_handlers:
            self.register(event_handler)

    def unregister_all(self):
        for event_handler in self._event_handlers:
            self.unregister(event_handler)
