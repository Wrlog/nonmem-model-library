"""The visual system every figure in this library is drawn with.

One palette, one type scale, one set of spacing rules, in a light and a dark
variant. The hues are the eight-slot colour-vision-safe set; slots are
assigned in fixed order and never cycled, and only the first three are used
where any series can sit beside any other, because those three are the ones
that clear the separation floors on the all-pairs test.

Everything a figure needs comes from here: `panel()` opens a figure with the
house margins, `title_block()` writes the title and its one-line subtitle,
`legend()` and `note()` place the two things that otherwise drift. A figure
module should not contain a hex code or a font size.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

DPI = 170

#: Preferred faces, best first. Whichever is installed wins; matplotlib's own
#: DejaVu Sans is the last resort so a bare machine still renders.
FONT_STACK = [
    "Inter", "Segoe UI", "Helvetica Neue", "Liberation Sans", "Arial",
    "DejaVu Sans",
]

# The type scale. Five sizes, and no figure may invent a sixth.
SIZE_TITLE = 12.0
SIZE_SUBTITLE = 9.8
SIZE_LABEL = 9.5
SIZE_TICK = 9.0
SIZE_NOTE = 8.6


@dataclass(frozen=True)
class Theme:
    """One mode's worth of colour. Roles, not hues, are what figures ask for."""

    name: str
    surface: str        # the plotting surface, matched to the card behind it
    plane: str          # the page the card sits on
    grid: str           # hairline gridlines, one shade off the surface
    border: str         # axis rules
    ink: str            # titles and anything that must be read first
    ink_2: str          # labels, legend text
    ink_3: str          # ticks, footnotes, reference lines
    series: tuple[str, ...]   # categorical slots, in fixed order
    ramp: tuple[str, ...]     # single-hue sequential ramp, light to dark
    good: str
    warning: str
    critical: str
    muted: str          # individual trajectories, and anything recessive
    #: Steps the ordinal ramp may draw from, in increasing order of value.
    #: Every step clears 2:1 against this mode's surface, which is what an
    #: ordered ramp needs and a sequential heatmap ramp does not.
    ordinal_steps: tuple[str, ...] = ()

    def ordinal(self, n: int) -> list[str]:
        """`n` steps of one hue for an *ordered* grouping.

        Dose levels and exposure tertiles are ordered, not nominal, so they
        take a ramp rather than categorical hues: the reader gets the order
        from the colour instead of having to learn an arbitrary mapping, and
        it keeps the categorical slots for series that really are identities.

        The steps are drawn evenly across the whole allowed range of the
        hue rather than from a fixed shortlist, which is what keeps adjacent
        steps far enough apart in lightness to be told apart at any `n` the
        library uses.
        """
        pool = self.ordinal_steps
        if n <= 1:
            return [pool[len(pool) // 2]]
        idx = [round(i * (len(pool) - 1) / (n - 1)) for i in range(n)]
        return [pool[i] for i in idx]


#: Status colours are fixed in both modes: they mean a state, never a series.
_GOOD, _WARNING, _CRITICAL = "#0ca30c", "#fab219", "#d03b3b"

LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    plane="#f9f9f7",
    grid="#e1e0d9",
    border="#c3c2b7",
    ink="#0b0b0b",
    ink_2="#52514e",
    ink_3="#898781",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
            "#008300", "#4a3aa7", "#e34948"),
    ramp=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab",
          "#104281"),
    good=_GOOD, warning=_WARNING, critical=_CRITICAL,
    muted="#c9c7bf",
    # Light mode: every blue step from 250 down, 250 being the lightest one
    # still at 2:1 against this surface.
    ordinal_steps=("#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                   "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"),
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    plane="#0d0d0d",
    grid="#2c2c2a",
    border="#383835",
    ink="#ffffff",
    ink_2="#c3c2b7",
    ink_3="#898781",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
            "#008300", "#9085e9", "#e66767"),
    ramp=("#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5",
          "#6da7ec"),
    good=_GOOD, warning=_WARNING, critical=_CRITICAL,
    muted="#4a4a46",
    # Dark mode runs the other way -- more value reads lighter on a dark
    # surface -- and stops at step 600, the darkest step still at 2:1.
    ordinal_steps=("#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5",
                   "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6",
                   "#cde2fb"),
)

THEMES = (LIGHT, DARK)


def _apply_rc() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.unicode_minus": False,
        "path.simplify": True,
        "path.simplify_threshold": 0.6,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.6,
        "legend.columnspacing": 1.4,
        "legend.labelspacing": 0.4,
    })


_apply_rc()


def panel(theme: Theme, nrows: int = 1, ncols: int = 1,
          width: float = 8.4, height: float = 4.3, **kwargs):
    """Open a figure on the theme's surface, with the house grid already set.

    Every figure in the library is the same width, so the page does not
    ripple as a reader scrolls through it.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), **kwargs)
    fig.set_facecolor(theme.surface)
    for ax in (axes.ravel() if hasattr(axes, "ravel") else [axes]):
        style_axes(ax, theme)
    return fig, axes


def style_axes(ax, theme: Theme, xgrid: bool = False, ygrid: bool = True) -> None:
    """House style: solid hairline grid, no box, no tick marks, ink labels."""
    ax.set_facecolor(theme.surface)
    ax.figure.set_facecolor(theme.surface)
    # matplotlib turns the grid *on* whenever line properties are passed, even
    # alongside False, so the properties only go in when a grid is wanted.
    if ygrid:
        ax.grid(True, axis="y", color=theme.grid, linewidth=0.7, linestyle="-")
    else:
        ax.grid(False, axis="y")
    if xgrid:
        ax.grid(True, axis="x", color=theme.grid, linewidth=0.7, linestyle="-")
    else:
        ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.border)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=theme.ink_3, labelsize=SIZE_TICK, length=0, pad=5)
    ax.xaxis.label.set_color(theme.ink_2)
    ax.yaxis.label.set_color(theme.ink_2)
    ax.xaxis.label.set_fontsize(SIZE_LABEL)
    ax.yaxis.label.set_fontsize(SIZE_LABEL)


def title_block(ax, theme: Theme, title: str, subtitle: str | None = None,
                small: bool = False) -> None:
    """Left-aligned title with an optional one-line subtitle beneath it.

    The subtitle carries the reading of the chart, so the caption underneath
    does not have to repeat what the axes already say.
    """
    size = SIZE_TITLE - (1.4 if small else 0)
    if subtitle:
        ax.set_title(title, fontsize=size, loc="left", color=theme.ink,
                     fontweight="600", pad=21)
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 7), textcoords="offset points",
                    ha="left", va="bottom",
                    fontsize=SIZE_SUBTITLE - (0.6 if small else 0),
                    color=theme.ink_3)
    else:
        ax.set_title(title, fontsize=size, loc="left", color=theme.ink,
                     fontweight="600", pad=8)


def legend(ax, theme: Theme, **kwargs):
    """A frameless legend in secondary ink; identity is never colour alone."""
    kwargs.setdefault("loc", "best")
    leg = ax.legend(frameon=False, fontsize=SIZE_LABEL, **kwargs)
    for text in leg.get_texts():
        text.set_color(theme.ink_2)
    return leg


def note(ax, theme: Theme, text: str, x: float = 0.99, y: float = 0.03,
         ha: str = "right", va: str = "bottom") -> None:
    """One short line inside the axes, for the thing the chart is evidence of."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=SIZE_NOTE, color=theme.ink_3)


def label_point(ax, theme: Theme, x, y, text: str, dx: float = 6,
                dy: float = 0, ha: str = "left", va: str = "center",
                colour: str | None = None) -> None:
    """A direct label on one mark. Used selectively, never on every point."""
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va=va, fontsize=SIZE_NOTE,
                color=colour or theme.ink_2)


def plain_log_ticks(ax, axis: str = "y") -> None:
    """Write log ticks as 0.1 / 1 / 10 rather than as powers of ten.

    A reader of a concentration axis wants the concentration, not its
    exponent.
    """
    from matplotlib.ticker import FuncFormatter

    fmt = FuncFormatter(lambda v, _: f"{v:g}" if v >= 0.01 else "")
    for name in axis:
        (ax.yaxis if name == "y" else ax.xaxis).set_major_formatter(fmt)


#: Palette size used when packing a figure into the page. These are line
#: drawings on a flat surface, so a palette holds them without visible loss
#: and roughly halves the bytes -- which matters when forty of them are
#: inlined into one self-contained page.
PALETTE_COLOURS = 192


def finish(fig, pad: float = 0.22) -> bytes:
    """Render to PNG bytes. Nothing in this package writes an image to disk."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight",
                pad_inches=pad, facecolor=fig.get_facecolor())
    plt.close(fig)

    image = Image.open(buf).convert("RGB")
    packed = image.quantize(colors=PALETTE_COLOURS,
                            method=Image.Quantize.MEDIANCUT,
                            dither=Image.Dither.NONE)
    out = io.BytesIO()
    packed.save(out, format="PNG", optimize=True)
    return out.getvalue()
