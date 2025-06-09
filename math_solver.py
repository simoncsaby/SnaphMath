import sympy as sp
from sympy import symbols, solve, expand, factor, simplify, latex
from sympy.parsing.sympy_parser import parse_expr
from sympy import solve_univariate_inequality, solveset, S, Abs
import re
from typing import Dict, List, Any
import warnings

# Opcionális numerikus támogatás
try:
    import numpy as np
    from scipy.optimize import fsolve, root
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy nem elérhető - numerikus megoldások korlátozott támogatással")

class MathSolver:
    """Matematikai egyenletek megoldója lépésenkénti levezetéssel - App.py kompatibilis verzió"""
    
    def __init__(self):
        """Inicializálás"""
        # Gyakori változók definiálása
        self.x, self.y, self.z = symbols('x y z')
        self.a, self.b, self.c = symbols('a b c')
        self.n, self.m, self.k = symbols('n m k')
        
        # Összes szimbólum egy dictionary-ben
        self.symbols_dict = {
            'x': self.x, 'y': self.y, 'z': self.z,
            'a': self.a, 'b': self.b, 'c': self.c,
            'n': self.n, 'm': self.m, 'k': self.k
        }
    
    def parse_equation(self, equation_str: str) -> Dict[str, Any]:
        """
        Egyenlet string feldolgozása és típusának meghatározása
        
        Args:
            equation_str: Az egyenlet string formában
            
        Returns:
            Dictionary az egyenlet adataival
        """
        try:
            # Tisztítás és formázás
            equation_str = self.clean_equation_string(equation_str)
            
            # Egyenlet típusának meghatározása
            eq_type = self.determine_equation_type(equation_str)
            
            # Parsing
            if '=' in equation_str:
                left, right = equation_str.split('=', 1)
                left_expr = parse_expr(left, transformations='all')
                right_expr = parse_expr(right, transformations='all')
                equation = sp.Eq(left_expr, right_expr)
            else:
                # Kifejezés egyszerűsítése vagy faktorizálása
                equation = parse_expr(equation_str, transformations='all')
            
            # Változók megkeresése
            variables = self.find_variables(equation)
            
            return {
                'success': True,
                'equation': equation,
                'type': eq_type,
                'variables': variables,
                'original_string': equation_str
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Parsing hiba: {str(e)}",
                'original_string': equation_str
            }
    
    def clean_equation_string(self, equation_str: str) -> str:
        """Egyenlet string tisztítása SymPy számára"""
        
        # Szóközök eltávolítása operátorok körül
        equation_str = re.sub(r'\s+', '', equation_str)
        
        # Implicit szorzás explicit szorzássá alakítása
        equation_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation_str)
        equation_str = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', equation_str)
        equation_str = re.sub(r'\)([a-zA-Z\d])', r')*\1', equation_str)
        equation_str = re.sub(r'([a-zA-Z\d])\(', r'\1*(', equation_str)
        
        # Hatványozás formázása
        equation_str = equation_str.replace('^', '**')
        
        # Gyakori függvények
        equation_str = equation_str.replace('sqrt', 'sqrt')
        
        # Abszolút érték
        equation_str = equation_str.replace('abs(', 'Abs(')
        
        # Egyenlőtlenség operátorok normalizálása
        equation_str = equation_str.replace('≥', '>=').replace('≤', '<=')
        
        return equation_str
    
    def determine_equation_type(self, equation_str: str) -> str:
        """Egyenlet típusának meghatározása"""
        
        # Egyenlőtlenségek
        if any(op in equation_str for op in ['>', '<', '>=', '<=']):
            return 'inequality'
        
        # Abszolút érték
        if 'Abs(' in equation_str or '|' in equation_str:
            return 'absolute_value'
        
        if '=' not in equation_str:
            return 'expression'
        
        # Lineáris egyenlet
        if '**2' not in equation_str and 'x**' not in equation_str and '^2' not in equation_str:
            return 'linear'
        
        # Másodfokú egyenlet
        if '**2' in equation_str or '^2' in equation_str:
            return 'quadratic'
        
        # Magasabb fokú
        if '**3' in equation_str or '**4' in equation_str:
            return 'polynomial'
        
        # Trigonometrikus
        if any(func in equation_str.lower() for func in ['sin', 'cos', 'tan']):
            return 'trigonometric'
        
        # Exponenciális/logaritmikus
        if any(func in equation_str.lower() for func in ['exp', 'log', 'ln']):
            return 'exponential'
        
        return 'general'
    
    def find_variables(self, equation) -> List[str]:
        """Egyenletben szereplő változók megkeresése"""
        if hasattr(equation, 'free_symbols'):
            return [str(var) for var in equation.free_symbols]
        return []
    
    def solve_step_by_step(self, equation_str: str) -> Dict[str, Any]:
        """
        Egyenlet lépésenkénti megoldása - fő API endpoint
        
        Args:
            equation_str: Az egyenlet string formában
            
        Returns:
            Dictionary a megoldással és lépésekkel
        """
        # Egyenlet parsing
        parsed = self.parse_equation(equation_str)
        
        if not parsed['success']:
            return parsed
        
        equation = parsed['equation']
        eq_type = parsed['type']
        variables = parsed['variables']
        
        try:
            # Megoldás típus szerint
            if eq_type == 'expression':
                return self.simplify_expression(equation, equation_str)
            elif eq_type == 'inequality':
                return self.solve_inequality(equation_str, equation, variables)
            elif eq_type == 'absolute_value':
                return self.solve_absolute_value(equation, variables)
            elif eq_type == 'linear':
                return self.solve_linear(equation, variables)
            elif eq_type == 'quadratic':
                return self.solve_quadratic(equation, variables)
            elif eq_type == 'polynomial':
                return self.solve_polynomial(equation, variables)
            elif eq_type == 'trigonometric':
                return self.solve_trigonometric(equation, variables)
            elif eq_type == 'exponential':
                return self.solve_exponential(equation, variables)
            else:
                return self.solve_general(equation, variables)
                
        except Exception as e:
            # Ha szimbolikus megoldás nem sikerül, próbáljuk numerikusan
            if SCIPY_AVAILABLE and '=' in equation_str:
                try:
                    numeric_result = self.solve_numerically(equation_str)
                    if numeric_result['success']:
                        return numeric_result
                except:
                    pass
            
            return {
                'success': False,
                'error': f"Megoldási hiba: {str(e)}",
                'equation_type': eq_type
            }
    
    def simplify_expression(self, expression, original_str: str) -> Dict[str, Any]:
        """Kifejezés egyszerűsítése lépésekkel"""
        
        steps = []
        steps.append({
            'step': 1,
            'description': 'Eredeti kifejezés',
            'expression': original_str,
            'latex': latex(expression)
        })
        
        # Kibontás
        expanded = expand(expression)
        if expanded != expression:
            steps.append({
                'step': len(steps) + 1,
                'description': 'Kifejezés kibontása',
                'expression': str(expanded),
                'latex': latex(expanded)
            })
        
        # Egyszerűsítés
        simplified = simplify(expanded)
        if simplified != expanded:
            steps.append({
                'step': len(steps) + 1,
                'description': 'Egyszerűsítés',
                'expression': str(simplified),
                'latex': latex(simplified)
            })
        
        # Faktorizálás próbálkozása
        try:
            factored = factor(simplified)
            if factored != simplified:
                steps.append({
                    'step': len(steps) + 1,
                    'description': 'Faktorizálás',
                    'expression': str(factored),
                    'latex': latex(factored)
                })
        except:
            pass
        
        return {
            'success': True,
            'type': 'simplification',
            'original': original_str,
            'result': str(simplified),
            'steps': steps
        }
    
    def solve_inequality(self, equation_str: str, equation, variables: List[str]) -> Dict[str, Any]:
        """Egyenlőtlenségek megoldása"""
        try:
            if not variables:
                return {'success': False, 'error': 'Nincs változó az egyenlőtlenségben'}
            
            var = symbols(variables[0])
            
            # Egyenlőtlenség típusának meghatározása
            if '>=' in equation_str:
                left, right = equation_str.split('>=')
                expr = parse_expr(left) - parse_expr(right)
                inequality = expr >= 0
            elif '<=' in equation_str:
                left, right = equation_str.split('<=')
                expr = parse_expr(left) - parse_expr(right)
                inequality = expr <= 0
            elif '>' in equation_str:
                left, right = equation_str.split('>')
                expr = parse_expr(left) - parse_expr(right)
                inequality = expr > 0
            elif '<' in equation_str:
                left, right = equation_str.split('<')
                expr = parse_expr(left) - parse_expr(right)
                inequality = expr < 0
            else:
                return {'success': False, 'error': 'Ismeretlen egyenlőtlenség operátor'}
            
            # Megoldás
            solution = solve_univariate_inequality(inequality, var, relational=False)
            
            steps = [
                {
                    'step': 1,
                    'description': 'Eredeti egyenlőtlenség',
                    'expression': equation_str,
                    'latex': latex(inequality)
                },
                {
                    'step': 2,
                    'description': 'Megoldás',
                    'expression': str(solution),
                    'latex': latex(solution)
                }
            ]
            
            return {
                'success': True,
                'type': 'inequality',
                'variable': str(var),
                'solutions': [str(solution)],
                'steps': steps
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Egyenlőtlenség megoldási hiba: {str(e)}"
            }
    
    def solve_absolute_value(self, equation, variables: List[str]) -> Dict[str, Any]:
        """Abszolút értékes egyenletek megoldása"""
        try:
            if not variables:
                return {'success': False, 'error': 'Nincs változó az egyenletben'}
            
            var = symbols(variables[0])
            solutions = solve(equation, var)
            
            steps = [
                {
                    'step': 1,
                    'description': 'Eredeti abszolút értékes egyenlet',
                    'expression': str(equation),
                    'latex': latex(equation)
                },
                {
                    'step': 2,
                    'description': 'Megoldások',
                    'expression': f"Megoldások: {[str(sol) for sol in solutions]}",
                    'latex': f"x = {', '.join([latex(sol) for sol in solutions])}"
                }
            ]
            
            return {
                'success': True,
                'type': 'absolute_value',
                'variable': str(var),
                'solutions': [str(sol) for sol in solutions],
                'steps': steps
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Abszolút értékes egyenlet hiba: {str(e)}"
            }
    
    def solve_linear(self, equation, variables: List[str]) -> Dict[str, Any]:
        """Lineáris egyenlet megoldása lépésekkel"""
        
        if not variables:
            return {'success': False, 'error': 'Nincs változó az egyenletben'}
        
        var = symbols(variables[0])
        steps = []
        
        steps.append({
            'step': 1,
            'description': 'Eredeti egyenlet',
            'expression': str(equation),
            'latex': latex(equation)
        })
        
        # Egyenlet átalakítása standard formára
        lhs = equation.lhs - equation.rhs
        steps.append({
            'step': 2,
            'description': 'Minden tag átmozgatása bal oldalra',
            'expression': f"{lhs} = 0",
            'latex': f"{latex(lhs)} = 0"
        })
        
        # Megoldás
        solutions = solve(equation, var)
        
        if solutions:
            var_latex = latex(var)
            sol_latex = latex(solutions[0])
            steps.append({
                'step': 3,
                'description': f'Megoldás {var} változóra',
                'expression': f"{var} = {solutions[0]}",
                'latex': f"{var_latex} = {sol_latex}"
            })
            
            # Ellenőrzés
            verification = equation.subs(var, solutions[0])
            verification_simplified = simplify(verification)
            verification_text = r'\text{Ellenőrzés: }'
            steps.append({
                'step': 4,
                'description': 'Ellenőrzés',
                'expression': f"Helyettesítés: {verification_simplified}",
                'latex': f"{verification_text} {latex(verification_simplified)}"
            })
        
        return {
            'success': True,
            'type': 'linear',
            'variable': str(var),
            'solutions': [str(sol) for sol in solutions],
            'steps': steps
        }
    
    def solve_quadratic(self, equation, variables: List[str]) -> Dict[str, Any]:
        """Másodfokú egyenlet megoldása lépésekkel"""
        
        if not variables:
            return {'success': False, 'error': 'Nincs változó az egyenletben'}
        
        var = symbols(variables[0])
        steps = []
        
        steps.append({
            'step': 1,
            'description': 'Eredeti egyenlet',
            'expression': str(equation),
            'latex': latex(equation)
        })
        
        # Standard forma: ax² + bx + c = 0
        lhs = equation.lhs - equation.rhs
        expanded = expand(lhs)
        
        steps.append({
            'step': 2,
            'description': 'Standard forma: ax² + bx + c = 0',
            'expression': f"{expanded} = 0",
            'latex': f"{latex(expanded)} = 0"
        })
        
        # Együtthatók meghatározása
        try:
            coeffs = sp.Poly(expanded, var).all_coeffs()
            if len(coeffs) >= 3:
                a, b, c = coeffs[:3]
                steps.append({
                    'step': 3,
                    'description': 'Együtthatók',
                    'expression': f"a = {a}, b = {b}, c = {c}",
                    'latex': f"a = {latex(a)}, b = {latex(b)}, c = {latex(c)}"
                })
                
                # Diszkrimináns
                discriminant = b**2 - 4*a*c
                delta_text = r'\Delta'
                steps.append({
                    'step': 4,
                    'description': 'Diszkrimináns',
                    'expression': f"Δ = b² - 4ac = {discriminant}",
                    'latex': f"{delta_text} = b^2 - 4ac = {latex(discriminant)}"
                })
        except:
            pass
        
        # Megoldás
        solutions = solve(equation, var)
        
        if len(solutions) == 2:
            var_latex = latex(var)
            sol1_latex = latex(solutions[0])
            sol2_latex = latex(solutions[1])
            steps.append({
                'step': len(steps) + 1,
                'description': 'Megoldások a képlet alapján',
                'expression': f"{var}₁ = {solutions[0]}, {var}₂ = {solutions[1]}",
                'latex': f"{var_latex}_1 = {sol1_latex}, {var_latex}_2 = {sol2_latex}"
            })
        elif len(solutions) == 1:
            var_latex = latex(var)
            sol_latex = latex(solutions[0])
            steps.append({
                'step': len(steps) + 1,
                'description': 'Megoldás',
                'expression': f"{var} = {solutions[0]}",
                'latex': f"{var_latex} = {sol_latex}"
            })
        else:
            no_solution_text = r'\text{Nincs valós megoldás}'
            steps.append({
                'step': len(steps) + 1,
                'description': 'Megoldás',
                'expression': "Nincs valós megoldás",
                'latex': no_solution_text
            })
        
        return {
            'success': True,
            'type': 'quadratic',
            'variable': str(var),
            'solutions': [str(sol) for sol in solutions],
            'steps': steps
        }
    
    def solve_polynomial(self, equation, variables: List[str]) -> Dict[str, Any]:
        """Magasabb fokú egyenlet megoldása"""
        
        var = symbols(variables[0])
        solutions = solve(equation, var)
        
        solutions_text = r'\text{Megoldások: }'
        solution_latexes = [latex(sol) for sol in solutions]
        steps = [
            {
                'step': 1,
                'description': 'Eredeti egyenlet',
                'expression': str(equation),
                'latex': latex(equation)
            },
            {
                'step': 2,
                'description': 'Megoldások',
                'expression': f"Megoldások: {[str(sol) for sol in solutions]}",
                'latex': f"{solutions_text} {', '.join(solution_latexes)}"
            }
        ]
        
        return {
            'success': True,
            'type': 'polynomial',
            'variable': str(var),
            'solutions': [str(sol) for sol in solutions],
            'steps': steps
        }
    
    def solve_trigonometric(self, equation, variables: List[str]) -> Dict[str, Any]:
        """Trigonometrikus egyenletek megoldása"""
        try:
            var = symbols(variables[0]) if variables else symbols('x')
            
            # Próbáljuk solveset-tel (jobb trigonometrikus támogatás)
            try:
                solutions = solveset(equation, var, domain=S.Reals)
                
                steps = [
                    {
                        'step': 1,
                        'description': 'Eredeti trigonometrikus egyenlet',
                        'expression': str(equation),
                        'latex': latex(equation)
                    },
                    {
                        'step': 2,
                        'description': 'Megoldások (általános forma)',
                        'expression': str(solutions),
                        'latex': latex(solutions)
                    }
                ]
                
                # Ha véges számú megoldás
                if solutions.is_finite_set:
                    solution_list = list(solutions)
                    return {
                        'success': True,
                        'type': 'trigonometric',
                        'variable': str(var),
                        'solutions': [str(sol) for sol in solution_list],
                        'steps': steps
                    }
                else:
                    # Végtelen megoldás (periodikus)
                    return {
                        'success': True,
                        'type': 'trigonometric',
                        'variable': str(var),
                        'solutions': [str(solutions)],
                        'general_solution': str(solutions),
                        'steps': steps
                    }
            except:
                # Ha solveset nem működik, próbáljuk a hagyományos solve-ot
                solutions = solve(equation, var)
                
                steps = [
                    {
                        'step': 1,
                        'description': 'Eredeti trigonometrikus egyenlet',
                        'expression': str(equation),
                        'latex': latex(equation)
                    },
                    {
                        'step': 2,
                        'description': 'Megoldások',
                        'expression': f"Megoldások: {[str(sol) for sol in solutions]}",
                        'latex': f"x = {', '.join([latex(sol) for sol in solutions])}"
                    }
                ]
                
                return {
                    'success': True,
                    'type': 'trigonometric',
                    'variable': str(var),
                    'solutions': [str(sol) for sol in solutions],
                    'steps': steps
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Trigonometrikus egyenlet hiba: {str(e)}"
            }
    
    def solve_exponential(self, equation, variables: List[str]) -> Dict[str, Any]:
        """Exponenciális/logaritmikus egyenletek megoldása"""
        try:
            var = symbols(variables[0]) if variables else symbols('x')
            solutions = solve(equation, var)
            
            steps = [
                {
                    'step': 1,
                    'description': 'Eredeti exponenciális/logaritmikus egyenlet',
                    'expression': str(equation),
                    'latex': latex(equation)
                },
                {
                    'step': 2,
                    'description': 'Megoldások',
                    'expression': f"Megoldások: {[str(sol) for sol in solutions]}",
                    'latex': f"x = {', '.join([latex(sol) for sol in solutions])}"
                }
            ]
            
            return {
                'success': True,
                'type': 'exponential',
                'variable': str(var),
                'solutions': [str(sol) for sol in solutions],
                'steps': steps
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Exponenciális egyenlet hiba: {str(e)}"
            }
    
    def solve_general(self, equation, variables: List[str]) -> Dict[str, Any]:
        """Általános egyenlet megoldása"""
        
        var = symbols(variables[0]) if variables else None
        solutions = solve(equation, var) if var else []
        
        steps = [
            {
                'step': 1,
                'description': 'Eredeti egyenlet',
                'expression': str(equation),
                'latex': latex(equation)
            }
        ]
        
        if solutions:
            solutions_text = r'\text{Megoldások: }'
            solution_latexes = [latex(sol) for sol in solutions]
            steps.append({
                'step': 2,
                'description': 'Megoldások',
                'expression': f"Megoldások: {[str(sol) for sol in solutions]}",
                'latex': f"{solutions_text} {', '.join(solution_latexes)}"
            })
        else:
            no_solution_text = r'\text{Nincs megoldás}'
            steps.append({
                'step': 2,
                'description': 'Nincs található megoldás',
                'expression': "Nincs megoldás vagy túl bonyolult",
                'latex': no_solution_text
            })
        
        return {
            'success': True,
            'type': 'general',
            'variable': str(var) if variables else None,
            'solutions': [str(sol) for sol in solutions],
            'steps': steps
        }
    
    def solve_numerically(self, equation_str: str, initial_guess=1.0) -> Dict[str, Any]:
        """Numerikus megoldás transcendens egyenletekhez (ha SciPy elérhető)"""
        if not SCIPY_AVAILABLE:
            return {
                'success': False,
                'error': 'Numerikus megoldás nem elérhető (SciPy nincs telepítve)'
            }
        
        try:
            # Egyenlet átalakítása f(x) = 0 formára
            if '=' in equation_str:
                left, right = equation_str.split('=', 1)
                expr_str = f"({left}) - ({right})"
            else:
                expr_str = equation_str
            
            # Python függvény létrehozása a SymPy kifejezésből
            expr = parse_expr(expr_str)
            var = list(expr.free_symbols)[0]
            
            # Lambdify a numerikus kiértékeléshez
            f = sp.lambdify(var, expr, 'numpy')
            
            # Numerikus megoldás
            solution = fsolve(f, initial_guess)[0]
            
            # Ellenőrzés
            check_value = f(solution)
            if abs(check_value) < 1e-10:
                steps = [
                    {
                        'step': 1,
                        'description': 'Eredeti egyenlet',
                        'expression': equation_str,
                        'latex': latex(expr) + " = 0"
                    },
                    {
                        'step': 2,
                        'description': f'Numerikus megoldás (kezdeti tipp: {initial_guess})',
                        'expression': f"{var} ≈ {solution:.6f}",
                        'latex': f"{latex(var)} \\approx {solution:.6f}"
                    },
                    {
                        'step': 3,
                        'description': 'Ellenőrzés',
                        'expression': f"f({solution:.6f}) = {check_value:.2e}",
                        'latex': f"f({solution:.6f}) = {check_value:.2e}"
                    }
                ]
                
                return {
                    'success': True,
                    'type': 'numerical',
                    'variable': str(var),
                    'solutions': [f"{solution:.6f}"],
                    'numerical_value': solution,
                    'steps': steps
                }
            else:
                return {
                    'success': False,
                    'error': f"Numerikus megoldás nem konvergált (f(x) = {check_value:.2e})"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Numerikus megoldás hiba: {str(e)}"
            }

# Használati példa és tesztelés
if __name__ == "__main__":
    solver = MathSolver()
    
    # Tesztelés különböző egyenletekkel
    # test_equations = [
    #     "2*x + 5 = 11",                    # lineáris
    #     "x**2 - 5*x + 6 = 0",             # másodfokú
    #     "x**3 - 8 = 0",                   # köbös
    #     "x**2 + 4*x + 4",                 # kifejezés faktorizáláshoz
    #     "sin(x) = 0.5",                   # trigonometrikus
    #     "exp(x) = 10",                    # exponenciális
    #     "abs(x - 3) = 2",                 # abszolút érték
    #     "x**2 - 5*x + 6 > 0",             # egyenlőtlenség
    # ]
    
    # for eq in test_equations:
    #     print(f"\n{'='*50}")
    #     print(f"Egyenlet: {eq}")
    #     print('='*50)
        
    #     result = solver.solve_step_by_step(eq)
        
    #     if result['success']:
    #         print(f"Típus: {result['type']}")
    #         if 'solutions' in result:
    #             print(f"Megoldások: {result['solutions']}")
            
    #         print("\nLépések:")
    #         for step in result['steps']:
    #             print(f"{step['step']}. {step['description']}")
    #             print(f"   {step['expression']}")
    #     else:
    #         print(f"Hiba: {result['error']}")