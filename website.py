from flask import Flask, request, render_template
from decimal import Decimal, getcontext
import math
import cmath
import random
import sympy as sp
import numpy as np

# Import functions from infilator.py
from infilator import (add, subtract, multiply, divide, modulus, factorial, exponentiate, sqrt,
                       sin, cos, tan, asin, acos, atan, log_base_10, natural_log, exponential,
                       abs_val, power_of_10, degrees_to_radians, radians_to_degrees, sinh, cosh,
                       tanh, arcsinh, arccosh, arctanh, nth_root, gcd, lcm, complex_operations,
                       permutations, combinations, random_number, matrix_multiply,
                       eigenvalues_eigenvectors, solve_linear_equations, numerical_integration,
                       numerical_differentiation, mean, median, standard_deviation)

# Set the precision to 500 digits
getcontext().prec = 500

app = Flask(__name__)

def format_result(result):
    result_str = str(result)
    if 'E' in result_str or len(result_str) > 100:
        return f"{result:.10E}"
    return result_str

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        num1 = request.form.get("num1")
        num2 = request.form.get("num2")
        operation = request.form.get("operation")
        
        if operation == "Addition":
            result = add(num1, num2)
        elif operation == "Subtraction":
            result = subtract(num1, num2)
        elif operation == "Multiplication":
            result = multiply(num1, num2)
        elif operation == "Division":
            result = divide(num1, num2)
        elif operation == "Modulus":
            result = modulus(num1, num2)
        elif operation == "Factorial":
            result = factorial(int(num1))
        elif operation == "Exponentiation":
            result = exponentiate(num1, num2)
        elif operation == "Square Root":
            result = sqrt(num1)
        elif operation == "Sine":
            result = sin(num1)
        elif operation == "Cosine":
            result = cos(num1)
        elif operation == "Tangent":
            result = tan(num1)
        elif operation == "Arcsine":
            result = asin(num1)
        elif operation == "Arccosine":
            result = acos(num1)
        elif operation == "Arctangent":
            result = atan(num1)
        elif operation == "Logarithm (Base 10)":
            result = log_base_10(num1)
        elif operation == "Natural Logarithm (ln)":
            result = natural_log(num1)
        elif operation == "Exponential Function (e^x)":
            result = exponential(num1)
        elif operation == "Absolute Value":
            result = abs_val(num1)
        elif operation == "Power of 10":
            result = power_of_10(num1)
        elif operation == "Degrees to Radians":
            result = degrees_to_radians(num1)
        elif operation == "Radians to Degrees":
            result = radians_to_degrees(num1)
        elif operation == "Hyperbolic Sine":
            result = sinh(num1)
        elif operation == "Hyperbolic Cosine":
            result = cosh(num1)
        elif operation == "Hyperbolic Tangent":
            result = tanh(num1)
        elif operation == "Inverse Hyperbolic Sine":
            result = arcsinh(num1)
        elif operation == "Inverse Hyperbolic Cosine":
            result = arccosh(num1)
        elif operation == "Inverse Hyperbolic Tangent":
            result = arctanh(num1)
        elif operation == "Nth Root":
            result = nth_root(num1, int(num2))
        elif operation == "GCD":
            result = gcd(int(num1), int(num2))
        elif operation == "LCM":
            result = lcm(int(num1), int(num2))
        elif operation == "Complex Number Operations":
            op = request.form.get("complex_op")
            a = num1
            b = num2
            result = complex_operations(op, a, b)
        elif operation == "Permutations":
            result = permutations(int(num1), int(num2))
        elif operation == "Combinations":
            result = combinations(int(num1), int(num2))
        elif operation == "Random Number Generation":
            result = random_number(float(num1), float(num2))
        elif operation == "Matrix Multiplication":
            A = [[int(x) for x in row.split()] for row in request.form.get("matrix_a").strip().split('\n')]
            B = [[int(x) for x in row.split()] for row in request.form.get("matrix_b").strip().split('\n')]
            result = matrix_multiply(A, B)
        elif operation == "Eigenvalues and Eigenvectors":
            matrix = [[int(x) for x in row.split()] for row in request.form.get("matrix").strip().split('\n')]
            eigenvalues, eigenvectors = eigenvalues_eigenvectors(matrix)
            result = f"Eigenvalues: {eigenvalues}\nEigenvectors: {eigenvectors}"
        elif operation == "Solve Linear Equations":
            A = [[int(x) for x in row.split()] for row in request.form.get("matrix_a").strip().split('\n')]
            B = [int(x) for x in request.form.get("constants").strip().split()]
            result = solve_linear_equations(A, B)
        elif operation == "Numerical Integration":
            f_expr = request.form.get("function")
            a = float(request.form.get("lower_bound"))
            b = float(request.form.get("upper_bound"))
            f = sp.lambdify(sp.Symbol('x'), sp.sympify(f_expr))
            result = numerical_integration(f, a, b, 1000)
        elif operation == "Numerical Differentiation":
            f_expr = request.form.get("function")
            x = float(request.form.get("point"))
            f = sp.lambdify(sp.Symbol('x'), sp.sympify(f_expr))
            result = numerical_differentiation(f, x)
        elif operation == "Statistical Functions":
            values = list(map(float, request.form.get("values").split()))
            result = f"Mean: {mean(values)}\nMedian: {median(values)}\nStandard Deviation: {standard_deviation(values)}"

        formatted_result = format_result(result)
        return render_template("index.html", result=formatted_result)

    return render_template("index.html", result="")

if __name__ == "__main__":
    app.run(debug=True)
