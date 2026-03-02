# ingkit Docstring Style Guide

This document defines the unified docstring rules for the ingkit package.  
The style follows **NumPy docstring format** and is optimized for:

- Human readability
- Scientific computing clarity
- Compatibility with Sphinx
- AI tools (ChatGPT, Codex, Copilot)
- Static analysis tools

---

# 1. General Style

- Use **NumPy docstring style**
- Write in clear, grammatically correct English
- Keep sentences concise
- Use periods at the end of sentences
- Line width: ~80 characters

---

# 2. Overall Structure

Use this section order:
```
"""
One-line summary.

Optional extended description.

Parameters
----------
...

Returns
-------
...

Raises
------
...

Notes
-----
...

See Also
--------
...

Examples
--------
...
"""
```
Only include sections that are necessary.

---

# 3. One-Line Summary

- Start with a verb (Compute, Apply, Return, Convert, Estimate…)
- End with a period
- Be precise and short

Example:

Compute the power spectral density of a signal.

---

# 4. Parameters Section

## 4.1 General Format
```
Parameters
----------
name : type, optional
    Description. Default is XXX.
...
```
Rules:

- Do NOT write default values in the type field
- Always write `, optional` for optional arguments
- Always write default in description:
  - `Default is 1.0.`
  - `Default is None.`
  - Strings must use `"double quotes"`

---

## 4.2 ndarray Shape Notation (Required Standard)

We adopt:

np.ndarray (n,)
np.ndarray (…, n)
np.ndarray (m, n)

Examples:
```
x : np.ndarray (n,)
    Input signal.

y : np.ndarray (..., n)
    Signal array. The last axis corresponds to time.
```

Rules:

- Shape must appear in the type field
- Axis meaning must be described in the description
- Use `(..., n)` when last axis is special

---

## 4.3 array_like

If array-like inputs are accepted:

```
x : array_like (n,)
    Input signal. Must be 1D.
```
---

## 4.4 Literal / Restricted Choices

Use set notation:
```

mode : {'linear', 'cubic'}, optional
    Interpolation mode. Default is 'linear'.
```

---

## 4.5 Units

Units must be written in parentheses:

```
fs : float
    Sampling frequency (Hz).
```

Never mix units into the type field.

---

# 5. Returns Section

Format:

```
name : type
    Description.
```
Rules:

- Include shape for ndarray
- Include units if relevant
- Keep concise

Example:
```
freqs : np.ndarray (m,)
    Frequency axis (Hz).

Pxx : np.ndarray (…, m)
    Power spectral density (unit^2/Hz).
```
---

# 6. Raises Section

## 6.1 General Format

Group by exception type:
```
Raises
------
ValueError
    If cutoff_freq is greater than the Nyquist frequency.
    If overlap_ratio is not in the interval [0, 1).
TypeError
    If input is not array-like.
```
Rules:

- Group multiple reasons under the same exception
- Use one sentence per condition
- Use "If ..." format
- Be explicit and precise
- Do not describe implementation internals

Correct:
```
If cutoff_freq is not in the interval (0, fs/2).
```

Incorrect:
```
If step <= 0.
```
---

# 7. Notes Section

Use only when necessary:

- Mathematical definitions
- Algorithmic details
- Physical meaning
- Equations

Example:
```
Notes
-----
The power spectral density is defined as
P(f) = |X(f)|^2 / (fs * U), 
where U = sum(w^2).
```
---

# 8. See Also

List related functions:
```
See Also
--------
scipy.signal.welch : Estimate power spectral density using Welch’s method.
```
---

# 9. Examples

Include only when helpful.

Use minimal, clear examples:
```
Examples
--------
freqs, Pxx = compute_psd(x, fs=1000)
```
---

# 10. Design Principles for ingkit

1. ndarray shape always specified.
2. Default values always written in description.
3. Units always written in parentheses.
4. Raises section included when validation exists.
5. No redundancy between signature and docstring.
6. Keep descriptions short but precise.
7. Prefer clarity over verbosity.

---

# 11. Template for ingkit Functions

```
def func(x, y=1.0):
    """
    One-line summary.

    Parameters
    ----------
    x : np.ndarray (n,)
        Description.
    y : float, optional
        Description. Default is 1.0.
    
    Returns
    -------
    out : np.ndarray (n,)
        Description.
    
    Raises
    ------
    ValueError
        If x has length less than 2.
    """
    ...
```
---

# End of Standard

This format must be used consistently across all modules in ingkit.