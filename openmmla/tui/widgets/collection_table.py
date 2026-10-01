"""the recorders of the Collection card as a table, like the Streams tab: one
row per recorder (one FFmpeg, one terminal tab), its cells dropdowns that open
under them. A recorder that writes several channels of its device (a Vimo
receiver on each, or 0,1) gets a row per channel right under its own, which
holds whose voice that channel is."""

from __future__ import annotations

from dataclasses import dataclass

from rich.cells import cell_len
from rich.text import Text
from textual import events
from textual.geometry import Region
from textual.message import Message
from textual.widgets import DataTable

from openmmla.collection.recording import audio_channel_selection

AUDIO_COLUMNS = ("#", "Host", "Device", "Channel", "Device Label", "Participant")
VIDEO_COLUMNS = ("#", "Host", "Device", "Device Label")

# what a dropdown cell ends on, as on the Streams tab
_ARROW = "  ▾"


def file_channels(device: object, channel: object, channels: object) -> list[int]:
    """the channels a recorder writes a file each for, when it writes several
    (each with a known count, or a list such as 0,1); [] for a recorder of one
    file: a downmix, one channel, or no device picked (it asks in its terminal)."""
    if not str(device or "").strip():
        return []
    try:
        count = int(str(channels or "").strip())
    except ValueError:
        count = 0
    try:
        picked = audio_channel_selection(str(channel or "").strip() or "mix", count if count > 0 else None)
    except ValueError:
        return []
    return [int(item) for item in picked] if len(picked) > 1 else []


def channel_picks(value: object, count: int) -> list[str]:
    """one Participant pick per channel out of a recorder's (5,group,none):
    none, or nothing, is bind later (""); as many as `count`."""
    parts = [part.strip() for part in str(value or "").split(",")]
    picks = ["" if part.lower() == "none" else part for part in parts]
    return (picks + [""] * count)[:count]


def joined_picks(picks: list[str]) -> str:
    """the Participant of a recorder of several channels: one pick per channel
    in channel order, none for one bound later; "" when every one is."""
    picks = [str(pick or "").strip() for pick in picks]
    if not any(picks):
        return ""
    return ",".join(pick or "none" for pick in picks)


def channel_text(channel: object) -> str:
    """a Channel pick as the table shows it: mix and each as they are, a
    channel as ch0, a list as ch0,ch2."""
    text = str(channel or "").strip() or "mix"
    parts = [part for part in text.replace(" ", ",").split(",") if part]
    if parts and all(part.isdigit() for part in parts):
        return ",".join(f"ch{part}" for part in parts)
    return text


@dataclass(frozen=True)
class TableRow:
    """one row of the table: recorder `index` (from 0), or one of the channels
    it writes (`channel`); the text of each cell, and the columns whose cell
    opens a dropdown."""

    index: int
    channel: int | None
    cells: tuple[str, ...]
    menus: frozenset[str] = frozenset()


class CollectionTable(DataTable):
    """the recorders of one tab of the Collection card. A click on a cell, or
    Enter on the cell under the cursor, asks for that cell's dropdown
    (CellPicked); what it offers and what a pick does is the launcher's."""

    # every row shows (a DataTable stops at the height of what holds it): the
    # card sits in a scroll of its own
    DEFAULT_CSS = """
    CollectionTable {
        height: auto;
        max-height: initial;
        margin: 0 0 1 0;
    }
    """

    class CellPicked(Message):
        """a cell was clicked, or Enter pressed on it: the role of the table,
        the recorder (from 0), the channel of a channel's row (None for the
        recorder's own) and the column's heading."""

        def __init__(self, table: CollectionTable, role: str, index: int, channel: int | None,
                     column: str) -> None:
            super().__init__()
            self.table = table
            self.role = role
            self.index = index
            self.channel = channel
            self.column = column

        @property
        def control(self) -> CollectionTable:
            return self.table

    def __init__(self, role: str, rows: list[TableRow] | None = None, **kwargs) -> None:
        super().__init__(cursor_type="cell", **kwargs)
        self.role = role
        # the rows to draw, and the rows drawn (what a row of the table is)
        # with the width of each column they make
        self._rows: list[TableRow] = list(rows or [])
        self._shown: list[TableRow] = []
        self._widths: list[int] = []

    @property
    def column_names(self) -> tuple[str, ...]:
        return AUDIO_COLUMNS if self.role == "audio" else VIDEO_COLUMNS

    @property
    def rows_shown(self) -> list[TableRow]:
        """the rows the table draws, top to bottom."""
        return list(self._rows)

    def on_mount(self) -> None:
        self._draw()

    def show(self, rows: list[TableRow]) -> None:
        """draw these rows in place of the ones shown; the cursor stays on
        its cell (its recorder or channel, and its column)."""
        self._rows = list(rows)
        if self.is_mounted:
            self._draw()

    def _draw(self) -> None:
        names = self.column_names
        kept = None
        if self.row_count:
            row, column = self.cursor_coordinate
            if 0 <= row < len(self._shown):
                kept = (self._shown[row].index, self._shown[row].channel, column)
        # the columns go too, so that each is as wide as what it holds now
        # (clear() alone keeps the widest it ever was); the cursor goes back
        # to the first cell
        self.clear(columns=True)
        self.add_columns(*names)
        self._shown = list(self._rows)
        # the arrows line up at the right edge of their column, under its heading
        widths = [
            max([cell_len(name) - cell_len(_ARROW)] + [cell_len(row.cells[column]) for row in self._shown
                                                       if column < len(row.cells)])
            for column, name in enumerate(names)
        ]
        drawn = [cell_len(name) for name in names]
        for row in self._shown:
            cells = []
            for column, name in enumerate(names):
                text = row.cells[column] if column < len(row.cells) else ""
                if name in row.menus:
                    cells.append(Text.assemble(text + " " * (widths[column] - cell_len(text)), (_ARROW, "dim")))
                else:
                    # plain text: a device label is no markup
                    cells.append(Text(text))
                drawn[column] = max(drawn[column], cells[-1].cell_len)
            self.add_row(*cells)
        # what the columns measure once the table has laid them out, known now
        # (a list may open under a cell before that)
        self._widths = drawn
        if kept is None or not self._shown:
            return
        index, channel, column = kept
        row = next((i for i, shown in enumerate(self._shown) if (shown.index, shown.channel) == (index, channel)),
                   None)
        if row is None:
            # a channel's row that went: its recorder's
            row = next((i for i, shown in enumerate(self._shown) if shown.index == index and shown.channel is None),
                       len(self._shown) - 1)
        self.move_cursor(row=row, column=min(column, len(names) - 1), scroll=False)

    def row_of(self, index: int, channel: int | None = None) -> int | None:
        """the row of recorder `index` (or of one of its channels) now."""
        return next((row for row, shown in enumerate(self._shown)
                     if shown.index == index and shown.channel == channel), None)

    def column_of(self, name: str) -> int | None:
        return self.column_names.index(name) if name in self.column_names else None

    def on_click(self, event: events.Click) -> None:
        # runs before DataTable's own handler, which moves the cursor to the cell
        meta = event.style.meta
        row, column = meta.get("row"), meta.get("column")
        if (not isinstance(row, int) or row < 0 or not isinstance(column, int) or column < 0
                or meta.get("out_of_bounds", False)):
            return
        self._ask(row, column)

    def action_select_cursor(self) -> None:
        super().action_select_cursor()
        if self.row_count:
            self._ask(*self.cursor_coordinate)

    def _ask(self, row: int, column: int) -> None:
        names = self.column_names
        if not 0 <= row < len(self._shown) or not 0 <= column < len(names):
            return
        shown = self._shown[row]
        self.post_message(self.CellPicked(self, self.role, shown.index, shown.channel, names[column]))

    def cell_region(self, row: int, column: int) -> Region:
        """where a cell is on the screen, for a list to open under it: by the
        widths of what was drawn, which the table lays out only once idle."""
        if not 0 <= column < len(self._widths):
            return Region(self.content_region.x, self.content_region.y, 0, 1)
        widths = [width + 2 * self.cell_padding for width in self._widths]
        y = (self.header_height if self.show_header else 0) + row
        area = self.content_region
        return Region(area.x + sum(widths[:column]) - round(self.scroll_x), area.y + y - round(self.scroll_y),
                      widths[column], 1)
