# CLEAR figure specification

Every figure script imports `style.py` and follows this. A figure that departs from it is wrong, whatever it shows.

**Size.** Double column 7.0 in wide; single column 3.4 in. Row height 2.4 in for a single row of panels; stacked panels 1.9 in each. Vector PDF plus 300 dpi PNG, same file stem.

**Type.** Helvetica or Arial; 8 pt axis labels and tick labels, 8 pt legend, 7 pt for any annotation, never smaller. Panel letters 10 pt bold, placed outside the axes at the top left of each panel by `st.letter(ax, "a")`, the same position in every figure.

**Palette (validated for colour-vision deficiency).** Functioning or reference series blue `#2a78d6`; collapsed or comparison series dark blue `#104281`; intervention or highlighted series orange `#eb6834`; a third series aqua `#1baf7a`; controls grey `#8a8985`. Regime fills: functioning `#b7d3f6`, two-state `#eb6834`, collapsed `#104281`. Text in ink `#0b0b0b` or muted ink `#52514e`, never in a series colour.

**Axes.** Demand inflow u always on a log axis, labelled "Demand inflow, u", ticks at 0.003, 0.01, 0.03, 0.1, 0.3; log scale is shown by the ticks, not stated in the label. Engagement s on [0, 1]. Thresholds are grey dotted verticals drawn by `st.thresholds(ax, u_up, u_rec)`, labelled "collapse" and "recovery" at the top of the topmost panel only. Grid: light horizontal lines only. Top and right spines off unless the panel is a filled map.

**Lines and markers.** Solid for stable branches and for arms with the feature; dashed for unstable branches or arms without it; dotted reserved for reference levels. Markers only where a quantity was measured at discrete points; filled means measured, hollow means the run left its branch. 1.4 pt lines, 4 pt markers.

**Text inside panels.** None beyond panel letters, threshold labels, direct series labels where a legend would collide, and at most one short pointer. Everything explanatory goes in the caption. Legends sit in empty plot space or outside the axes, never over data.

**Numbers.** Each script writes a `<stem>_numbers.json` with every value the caption quotes, at the reference set (R = 2, g = 2.5, n = 4, C = 1, E = 1, k* = 1, leaks 0.1, κ = 1), slow layer off unless the caption says otherwise.
