## 2024-05-24 - Move sklearn tag imports to module level
**Learning:** Method-level imports violate package hygiene. Backward compatibility can be maintained with try-except blocks at the module level.
**Action:** Always place dependency imports at the top of the file, wrapped in try-except if they are version-dependent.
