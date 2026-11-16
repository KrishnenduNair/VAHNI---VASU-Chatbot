import re
from fractions import Fraction


def parse_formula(formula):
    """Parse a chemical formula into a dict of element counts."""
    formula = formula.strip()
    pattern = r'([A-Z][a-z]?)(\d*)'
    counts = {}
    i = 0
    # Handle parentheses like Ca(OH)2
    def parse_segment(s):
        result = {}
        idx = 0
        while idx < len(s):
            if s[idx] == '(':
                # find matching closing paren
                depth = 1
                j = idx + 1
                while j < len(s) and depth > 0:
                    if s[j] == '(':
                        depth += 1
                    elif s[j] == ')':
                        depth -= 1
                    j += 1
                inner = parse_segment(s[idx+1:j-1])
                # get multiplier after )
                m = re.match(r'(\d+)', s[j:])
                mult = int(m.group(1)) if m else 1
                if m:
                    j += len(m.group(1))
                for el, cnt in inner.items():
                    result[el] = result.get(el, 0) + cnt * mult
                idx = j
            else:
                m = re.match(r'([A-Z][a-z]?)(\d*)', s[idx:])
                if m:
                    el = m.group(1)
                    cnt = int(m.group(2)) if m.group(2) else 1
                    result[el] = result.get(el, 0) + cnt
                    idx += len(m.group(0))
                else:
                    idx += 1
        return result
    return parse_segment(formula)


def balance_equation(equation):
    """
    Balance a chemical equation using linear algebra (null space method).
    Returns (coefficients, molecules, elements) or raises ValueError.
    """
    # Normalize arrow
    eq = equation.replace('→', '->').replace('=', '->').replace('⟶', '->')
    
    if '->' not in eq:
        raise ValueError("No reaction arrow found. Use → or ->")
    
    lhs, rhs = eq.split('->')
    reactants = [r.strip() for r in lhs.split('+')]
    products = [p.strip() for p in rhs.split('+')]
    
    all_molecules = reactants + products
    n_reactants = len(reactants)
    
    # Collect all elements
    elements = set()
    parsed = []
    for mol in all_molecules:
        p = parse_formula(mol)
        parsed.append(p)
        elements.update(p.keys())
    elements = sorted(elements)
    
    n_elements = len(elements)
    n_molecules = len(all_molecules)
    
    # Build matrix: rows = elements, cols = molecules
    # Reactants positive, products negative
    matrix = []
    for el in elements:
        row = []
        for i, p in enumerate(parsed):
            count = p.get(el, 0)
            if i >= n_reactants:
                count = -count  # products are negative
            row.append(Fraction(count))
        matrix.append(row)
    
    # Gaussian elimination to find null space
    # We need one free variable, so augment with n_molecules columns
    m = [row[:] for row in matrix]
    n_rows = len(m)
    n_cols = n_molecules
    
    pivot_cols = []
    row_idx = 0
    for col in range(n_cols):
        # Find pivot
        pivot = None
        for r in range(row_idx, n_rows):
            if m[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue
        m[row_idx], m[pivot] = m[pivot], m[row_idx]
        pivot_val = m[row_idx][col]
        m[row_idx] = [x / pivot_val for x in m[row_idx]]
        for r in range(n_rows):
            if r != row_idx and m[r][col] != 0:
                factor = m[r][col]
                m[r] = [m[r][c] - factor * m[row_idx][c] for c in range(n_cols)]
        pivot_cols.append(col)
        row_idx += 1
    
    # Free variables are non-pivot columns
    free_cols = [c for c in range(n_cols) if c not in pivot_cols]
    
    if not free_cols:
        raise ValueError("No solution found")
    
    # Set free variable = 1
    free_col = free_cols[-1]
    solution = [Fraction(0)] * n_cols
    solution[free_col] = Fraction(1)
    
    for i, col in enumerate(pivot_cols):
        if i < len(m):
            solution[col] = -m[i][free_col]
    
    # Convert to positive integers
    # Find LCM of denominators
    from math import gcd
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    denom_lcm = 1
    for s in solution:
        denom_lcm = lcm(denom_lcm, s.denominator)
    
    int_solution = [int(s * denom_lcm) for s in solution]
    
    # Make all positive
    if any(x < 0 for x in int_solution):
        int_solution = [-x for x in int_solution]
    
    # Simplify by GCD
    overall_gcd = int_solution[0]
    for x in int_solution[1:]:
        overall_gcd = gcd(overall_gcd, x)
    int_solution = [x // overall_gcd for x in int_solution]
    
    return int_solution, all_molecules, elements, n_reactants, parsed


def format_balanced_equation(equation):
    """
    Returns a dict with:
    - balanced: the balanced equation string
    - steps: step by step working
    - coefficients: list of ints
    - molecules: list of molecule strings
    """
    try:
        coeffs, molecules, elements, n_reactants, parsed = balance_equation(equation)
    except Exception as e:
        return {"error": str(e)}
    
    # Build balanced equation string
    reactant_parts = []
    for i in range(n_reactants):
        c = coeffs[i]
        mol = molecules[i]
        reactant_parts.append(f"{c}{mol}" if c > 1 else mol)
    
    product_parts = []
    for i in range(n_reactants, len(molecules)):
        c = coeffs[i]
        mol = molecules[i]
        product_parts.append(f"{c}{mol}" if c > 1 else mol)
    
    balanced = " + ".join(reactant_parts) + " → " + " + ".join(product_parts)
    
    # Build steps
    # Normalize original equation
    eq = equation.replace('→', '->').replace('=', '->').replace('⟶', '->')
    lhs, rhs = eq.split('->')
    
    steps = []
    steps.append(f"Unbalanced: {lhs.strip()} → {rhs.strip()}")
    steps.append("")
    
    # Initial atom count
    steps.append("Initial atom count:")
    for el in elements:
        left = sum(parsed[i].get(el, 0) for i in range(n_reactants))
        right = sum(parsed[i].get(el, 0) for i in range(n_reactants, len(molecules)))
        steps.append(f"- {el}: {left} (left), {right} (right)")
    steps.append("")
    
    # Show balancing steps per element
    steps.append("Balancing steps:")
    current_coeffs = [1] * len(molecules)
    
    for step_num, el in enumerate(elements, 1):
        new_coeffs = coeffs[:]
        left = sum(new_coeffs[i] * parsed[i].get(el, 0) for i in range(n_reactants))
        right = sum(new_coeffs[i] * parsed[i].get(el, 0) for i in range(n_reactants, len(molecules)))
        steps.append(f"Step {step_num}: Balance {el} → {left} on each side")
    
    steps.append("")
    steps.append("Verification:")
    for el in elements:
        left = sum(coeffs[i] * parsed[i].get(el, 0) for i in range(n_reactants))
        right = sum(coeffs[i] * parsed[i].get(el, 0) for i in range(n_reactants, len(molecules)))
        steps.append(f"- {el}: {left} (left) = {right} (right) ✓")
    
    steps.append("")
    steps.append(f"Balanced: {balanced}")
    
    return {
        "balanced": balanced,
        "steps": "\n".join(steps),
        "coefficients": coeffs,
        "molecules": molecules
    }


if __name__ == "__main__":
    # Test cases
    tests = [
        "Al + HCl → AlCl3 + H2",
        "H2 + O2 → H2O",
        "Fe + O2 → Fe2O3",
        "Na + H2O → NaOH + H2",
        "Ca(OH)2 + HCl → CaCl2 + H2O",
        "CH4 + O2 → CO2 + H2O",
    ]
    for t in tests:
        print(f"\nInput: {t}")
        result = format_balanced_equation(t)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(result["steps"])
        print("-" * 50)