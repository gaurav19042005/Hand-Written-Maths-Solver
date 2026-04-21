"""
solver.py
Turns a list of predicted symbols into a math expression and evaluates it.
Supports: arithmetic  (+ - * /)  with multi-digit numbers.
"""

import re
from sympy import sympify, SympifyError


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def symbols_to_expression(symbols: list[str]) -> str:
    """
    Join individual symbol strings into a valid math expression string.
    Adjacent digits are merged into multi-digit numbers.
    Trailing '=' is stripped.
    """
    # Strip any '=' at the end
    syms = [s for s in symbols if s.strip() not in ('=', '')]
    return "".join(syms)


def evaluate(expr_str: str) -> str:
    """
    Evaluate a math expression string.
    Returns a human-readable result string or an error message.

    Supports:
      - Arithmetic: 3+5*2, 12/4, 100-37
      - Multi-digit numbers: 25*4
      - Parentheses (if segmentation detects them)
    """
    if not expr_str:
        return "Error: empty expression"

    # Safety: allow only digits, operators, spaces, dots, parentheses
    clean = re.sub(r'[^0-9+\-*/().\s]', '', expr_str)
    clean = clean.strip()

    if not clean:
        return "Error: no valid expression found"

    try:
        result = sympify(clean)
        # Return integer if result is whole, else decimal
        if result == int(result):
            return str(int(result))
        else:
            return str(float(result))
    except (SympifyError, ZeroDivisionError) as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"


def solve_from_symbols(symbols: list[str]) -> dict:
    """
    Full pipeline: symbols → expression string → answer.
    Returns dict with keys: expression, answer, error (bool).
    """
    expr = symbols_to_expression(symbols)
    answer = evaluate(expr)
    has_error = answer.startswith("Error")
    return {
        "expression": expr,
        "answer":     answer,
        "error":      has_error,
    }


# ─── CLI DEMO ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ["3", "+", "5"],
        ["1", "2", "*", "4"],
        ["1", "0", "0", "-", "3", "7"],
        ["8", "/", "2"],
        ["2", "+", "3", "*", "4"],        # should give 14 (BODMAS)
        ["1", "0", "/", "0"],             # division by zero
    ]
    for syms in tests:
        res = solve_from_symbols(syms)
        status = "OK" if not res["error"] else "ERR"
        print(f"[{status}]  {res['expression']}  =  {res['answer']}")
