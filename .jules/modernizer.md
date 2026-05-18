## YYYY-MM-DD - Use set literals for constants
**Learning:** `asgl.constants` defines static collections like `INDIV_NONADAPTIVE`, `GROUP_ADAPTIVE`, etc. as lists. These are frequently used for membership tests like `if self.penalization in (INDIV_ADAPTIVE + GROUP_ADAPTIVE):`.
**Action:** Convert constant arrays defined in `constants.py` to `set` for faster O(1) lookups, and use set union (or pre-combine) for combinations. Set lookups improve performance for O(1) membership checks compared to list traversal and concatenation.
