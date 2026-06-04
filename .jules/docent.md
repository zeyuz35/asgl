## 2024-06-04 - Fix docstring default parameter formatting
**Learning:** Parameter defaults in docstrings should use consistent no-space formatting (`default=value`) and typos (like `defaul=`) undermine professional presentation.
**Action:** Use regex searches (e.g. `grep -rnE "(default =|defaul=)"`) to ensure consistent parameter default formatting in docstrings across the codebase.
