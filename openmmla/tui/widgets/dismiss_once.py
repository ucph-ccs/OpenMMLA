"""a modal screen that is closed once.

A double click on a dropdown's option (or on a button that closes a screen),
or a click and a key, can put two closes on one screen's queue before the
first takes it off the stack. Textual then raises ScreenStackError for the
second, and the console stops: picking `yes` in a stream's Record dropdown
did that. The first close counts; a later one is left alone."""

from __future__ import annotations


class DismissOnce:
    """mixed in before ModalScreen: dismiss() closes the screen the first time
    only, and returns None afterwards."""

    _dismissed = False

    def dismiss(self, result=None):
        if self._dismissed:
            return None
        self._dismissed = True
        return super().dismiss(result)
