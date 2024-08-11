from decimal import Decimal, getcontext
import math
import cmath
import random
import numpy as np

# Set the precision to 100 digits (or more if needed)
getcontext().prec = 50040

def format_result(result):
    result_str = str(result)
    if 'E' in result_str or len(result_str) > 100:
        choice = input("The result is large. Do you want to see the full result? (y/n): ")
        if choice.lower() == 'y':
            return f"{result.normalize()}"
        else:
            return f"{result:.10E}"
    return result_str

# Basic arithmetic operations
def add(x, y):
    return Decimal(x) + Decimal(y)

def subtract(x, y):
    return Decimal(x) - Decimal(y)

def multiply(x, y):
    return Decimal(x) * Decimal(y)

def divide(x, y):
    if Decimal(y) == 0:
        return "Error: Division by zero"
    return Decimal(x) / Decimal(y)

def modulus(x, y):
    return Decimal(x) % Decimal(y)

# Factorial
def factorial(n):
    if n == 0:
        return 1
    result = Decimal(1)
    for i in range(1, n + 1):
        result *= Decimal(i)
    return result

# Exponentiation and square root
def exponentiate(base, exp):
    return Decimal(base) ** Decimal(exp)

def sqrt(x):
    if Decimal(x) < 0:
        return "Error: Square root of a negative number"
    return Decimal(x).sqrt()

# Trigonometric functions
def sin(x):
    return Decimal(math.sin(math.radians(float(x))))

def cos(x):
    return Decimal(math.cos(math.radians(float(x))))

def tan(x):
    return Decimal(math.tan(math.radians(float(x))))

def asin(x):
    return Decimal(math.degrees(math.asin(float(x))))

def acos(x):
    return Decimal(math.degrees(math.acos(float(x))))

def atan(x):
    return Decimal(math.degrees(math.atan(float(x))))

# Logarithms and exponential functions
def log_base_10(x):
    return Decimal(math.log10(float(x)))

def natural_log(x):
    return Decimal(math.log(float(x)))

def exponential(x):
    return Decimal(math.exp(float(x)))

# Absolute value and power of 10
def abs_val(x):
    return Decimal(abs(float(x)))

def power_of_10(x):
    return Decimal(10) ** Decimal(x)

# Conversion between degrees and radians
def degrees_to_radians(x):
    return Decimal(math.radians(float(x)))

def radians_to_degrees(x):
    return Decimal(math.degrees(float(x)))

# Hyperbolic functions
def sinh(x):
    return Decimal(math.sinh(float(x)))

def cosh(x):
    return Decimal(math.cosh(float(x)))

def tanh(x):
    return Decimal(math.tanh(float(x)))

def arcsinh(x):
    return Decimal(math.asinh(float(x)))

def arccosh(x):
    return Decimal(math.acosh(float(x)))

def arctanh(x):
    return Decimal(math.atanh(float(x)))

# Root calculation
def nth_root(x, n):
    return Decimal(x) ** (Decimal(1) / Decimal(n))

# GCD and LCM
def gcd(x, y):
    return math.gcd(int(x), int(y))

def lcm(x, y):
    return abs(int(x) * int(y)) // math.gcd(int(x), int(y))

# Complex number operations
def complex_operations(op, a, b):
    a = complex(a)
    b = complex(b)
    if op == 'add':
        return a + b
    elif op == 'subtract':
        return a - b
    elif op == 'multiply':
        return a * b
    elif op == 'divide':
        return a / b

# Permutations and Combinations
def permutations(n, r):
    return factorial(n) / factorial(n - r)

def combinations(n, r):
    return factorial(n) / (factorial(r) * factorial(n - r))

# Random number generation
def random_number(low, high):
    return Decimal(random.uniform(low, high))

# Matrix operations (example: multiplication)
def matrix_multiply(A, B):
    if len(A[0]) != len(B):
        return "Error: Incompatible matrix dimensions for multiplication"
    result = [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]
    return result


# Eigenvalues and Eigenvectors
def eigenvalues_eigenvectors(matrix):
    import numpy as np
    matrix = np.array(matrix)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues.tolist(), eigenvectors.tolist()

# Solving Linear Equations
def solve_linear_equations(A, B):
    import numpy as np
    A = np.array(A)
    B = np.array(B)
    return np.linalg.solve(A, B).tolist()

# Integration and Differentiation (Numerical Methods)
def numerical_integration(f, a, b, n):
    # Trapezoidal rule
    h = (b - a) / n
    integral = (f(a) + f(b)) / 2
    for i in range(1, n):
        integral += f(a + i * h)
    integral *= h
    return Decimal(integral)

def numerical_differentiation(f, x, h=1e-5):
    return Decimal((f(x + h) - f(x - h)) / (2 * h))

# Statistical Functions
def mean(values):
    return Decimal(sum(values)) / Decimal(len(values))

def median(values):
    sorted_values = sorted(values)
    n = len(values)
    if n % 2 == 1:
        return Decimal(sorted_values[n // 2])
    else:
        return Decimal((sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2)


def standard_deviation(values):
    values = [Decimal(x) for x in values]  # Convert all values to Decimal
    mu = mean(values)
    variance = sum((x - mu) ** 2 for x in values) / len(values)
    return Decimal(math.sqrt(float(variance)))  # Convert variance to float for sqrt


def sqrt(x):
    try:
        x = Decimal(x)
        if x < 0:
            return "Error: Square root of a negative number"
        return x.sqrt()
    except Exception as e:
        return f"Error: {str(e)}"

def calculator():
    print("Welcome to Infilater: The High-Precision Calculator!")
    
    while True:
        print("\nChoose an operation:")
        print("1: Addition")
        print("2: Subtraction")
        print("3: Multiplication")
        print("4: Division")
        print("5: Factorial")
        print("6: Exponentiation")
        print("7: Square Root")
        print("8: Modulus")
        print("9: Sine")
        print("10: Cosine")
        print("11: Tangent")
        print("12: Arcsine")
        print("13: Arccosine")
        print("14: Arctangent")
        print("15: Logarithm (Base 10)")
        print("16: Natural Logarithm (ln)")
        print("17: Exponential Function (e^x)")
        print("18: Absolute Value")
        print("19: Power of 10")
        print("20: Degrees to Radians")
        print("21: Radians to Degrees")
        print("22: Hyperbolic Sine")
        print("23: Hyperbolic Cosine")
        print("24: Hyperbolic Tangent")
        print("25: Inverse Hyperbolic Sine")
        print("26: Inverse Hyperbolic Cosine")
        print("27: Inverse Hyperbolic Tangent")
        print("28: Nth Root")
        print("29: GCD")
        print("30: LCM")
        print("31: Complex Number Operations")
        print("32: Permutations")
        print("33: Combinations")
        print("34: Random Number Generation")
        print("35: Matrix Multiplication")
        print("36: Eigenvalues and Eigenvectors")
        print("37: Solve Linear Equations")
        print("38: Numerical Integration")
        print("39: Numerical Differentiation")
        print("40: Statistical Functions (Mean, Median, Standard Deviation)")
        print("41: Exit")
        
        choice = input("Enter your choice (1-41): ")
        
        if choice == '41':
            print("Exiting Infilater. Goodbye!")
            break
        
        if choice in ['1', '2', '3', '4', '6', '8']:
            num1 = input("Enter the first number: ")
            num2 = input("Enter the second number: ")
            
            if choice == '1':
                result = add(num1, num2)
            elif choice == '2':
                result = subtract(num1, num2)
            elif choice == '3':
                result = multiply(num1, num2)
            elif choice == '4':
                result = divide(num1, num2)
            elif choice == '6':
                result = exponentiate(num1, num2)
            elif choice == '8':
                result = modulus(num1, num2)
            
            print("Result:", format_result(result))
        
        elif choice == '5':
            num = int(input("Enter the number for factorial: "))
            result = factorial(num)
            print("Result:", format_result(result))
        
        elif choice == '7':
            num = input("Enter the number for square root: ")
            result = sqrt(num)
            print("Result:", format_result(result))
        
        elif choice in ['9', '10', '11', '12', '13', '14']:
            num = input("Enter the number (in degrees for trig functions): ")
            
            if choice == '9':
                result = sin(num)
            elif choice == '10':
                result = cos(num)
            elif choice == '11':
                result = tan(num)
            elif choice == '12':
                result = asin(num)
            elif choice == '13':
                result = acos(num)
            elif choice == '14':
                result = atan(num)
            
            print("Result:", format_result(result))
        
        elif choice == '15':
            num = input("Enter the number for logarithm (Base 10): ")
            result = log_base_10(num)
            print("Result:", format_result(result))
        
        elif choice == '16':
            num = input("Enter the number for natural logarithm (ln): ")
            result = natural_log(num)
            print("Result:", format_result(result))
        
        elif choice == '17':
            num = input("Enter the number for exponential function (e^x): ")
            result = exponential(num)
            print("Result:", format_result(result))
        
        elif choice == '18':
            num = input("Enter the number for absolute value: ")
            result = abs_val(num)
            print("Result:", format_result(result))
        
        elif choice == '19':
            num = input("Enter the exponent for power of 10: ")
            result = power_of_10(num)
            print("Result:", format_result(result))
        
        elif choice == '20':
            num = input("Enter the angle in degrees to convert to radians: ")
            result = degrees_to_radians(num)
            print("Result:", format_result(result))
        
        elif choice == '21':
            num = input("Enter the angle in radians to convert to degrees: ")
            result = radians_to_degrees(num)
            print("Result:", format_result(result))
        
        elif choice == '22':
            num = input("Enter the number for hyperbolic sine (sinh): ")
            result = sinh(num)
            print("Result:", format_result(result))
        
        elif choice == '23':
            num = input("Enter the number for hyperbolic cosine (cosh): ")
            result = cosh(num)
            print("Result:", format_result(result))
        
        elif choice == '24':
            num = input("Enter the number for hyperbolic tangent (tanh): ")
            result = tanh(num)
            print("Result:", format_result(result))
        
        elif choice == '25':
            num = input("Enter the number for inverse hyperbolic sine (arcsinh): ")
            result = arcsinh(num)
            print("Result:", format_result(result))
        
        elif choice == '26':
            num = input("Enter the number for inverse hyperbolic cosine (arccosh): ")
            result = arccosh(num)
            print("Result:", format_result(result))
        
        elif choice == '27':
            num = input("Enter the number for inverse hyperbolic tangent (arctanh): ")
            result = arctanh(num)
            print("Result:", format_result(result))
        
        elif choice == '28':
            num = input("Enter the number for nth root calculation: ")
            n = int(input("Enter the root value (n): "))
            result = nth_root(num, n)
            print("Result:", format_result(result))
        
        elif choice == '29':
            num1 = int(input("Enter the first number for GCD: "))
            num2 = int(input("Enter the second number for GCD: "))
            result = gcd(num1, num2)
            print("Result:", format_result(result))
        
        elif choice == '30':
            num1 = int(input("Enter the first number for LCM: "))
            num2 = int(input("Enter the second number for LCM: "))
            result = lcm(num1, num2)
            print("Result:", format_result(result))
        
        elif choice == '31':
            op = input("Enter the operation (add, subtract, multiply, divide): ")
            a = input("Enter the first complex number (a + bj): ")
            b = input("Enter the second complex number (c + dj): ")
            result = complex_operations(op, a, b)
            print("Result:", format_result(result))
        
        elif choice == '32':
            n = int(input("Enter the total number of items (n): "))
            r = int(input("Enter the number of items to choose (r): "))
            result = permutations(n, r)
            print("Result:", format_result(result))
        
        elif choice == '33':
            n = int(input("Enter the total number of items (n): "))
            r = int(input("Enter the number of items to choose (r): "))
            result = combinations(n, r)
            print("Result:", format_result(result))
        
        elif choice == '34':
            low = float(input("Enter the lower bound: "))
            high = float(input("Enter the upper bound: "))
            result = random_number(low, high)
            print("Result:", format_result(result))
        
        elif choice == '35':
            A = [[int(x) for x in input("Enter the first matrix row (space-separated): ").split()]]
            B = [[int(x) for x in input("Enter the second matrix row (space-separated): ").split()]]
            result = matrix_multiply(A, B)
            print("Result:", result)
        
        elif choice == '36':
            matrix = [[int(x) for x in input("Enter the matrix row (space-separated): ").split()] for _ in range(int(input("Enter number of rows: ")))]
            eigenvalues, eigenvectors = eigenvalues_eigenvectors(matrix)
            print("Eigenvalues:", eigenvalues)
            print("Eigenvectors:", eigenvectors)
        
        elif choice == '37':
            A = [[int(x) for x in input("Enter the coefficient matrix row (space-separated): ").split()] for _ in range(int(input("Enter number of rows: ")))]
            B = [int(x) for x in input("Enter the constants (space-separated): ").split()]
            result = solve_linear_equations(A, B)
            print("Solution:", result)
        
        elif choice == '38':
            import sympy as sp
            f_expr = input("Enter the function to integrate (in terms of x): ")
            a = float(input("Enter the lower bound: "))
            b = float(input("Enter the upper bound: "))
            f = sp.lambdify(sp.Symbol('x'), sp.sympify(f_expr))
            result = numerical_integration(f, a, b, 1000)
            print("Result:", format_result(result))
        
        elif choice == '39':
            import sympy as sp
            f_expr = input("Enter the function to differentiate (in terms of x): ")
            x = float(input("Enter the point to evaluate the derivative at: "))
            f = sp.lambdify(sp.Symbol('x'), sp.sympify(f_expr))
            result = numerical_differentiation(f, x)
            print("Result:", format_result(result))
        
        elif choice == '40':
            values = list(map(float, input("Enter the list of numbers (space-separated): ").split()))
            print("Mean:", format_result(mean(values)))
            print("Median:", format_result(median(values)))
            print("Standard Deviation:", format_result(standard_deviation(values)))
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    calculator()