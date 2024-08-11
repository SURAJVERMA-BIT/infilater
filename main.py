from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from decimal import Decimal, getcontext
import math
import random
from matplotlib.image import QUADRIC
import numpy as np  # For matrix operations
from scipy import linalg
from sympy import Derivative  # For eigenvalues and eigenvectors

# Set the precision to 500 digits
getcontext().prec = 500

def format_result(result):
    result_str = str(result)
    if 'E' in result_str or len(result_str) > 100:
        return f"{result:.10E}"
    return result_str

# Define your functions here
def add(a, b):
    return Decimal(a) + Decimal(b)

def subtract(a, b):
    return Decimal(a) - Decimal(b)

def multiply(a, b):
    return Decimal(a) * Decimal(b)

def divide(a, b):
    return Decimal(a) / Decimal(b)

def modulus(a, b):
    return Decimal(a) % Decimal(b)

def factorial(a):
    return math.factorial(int(a))

def exponentiation(a, b):
    return Decimal(a) ** Decimal(b)

def square_root(a):
    return Decimal(a).sqrt()

def sine(a):
    return math.sin(math.radians(float(a)))

def cosine(a):
    return math.cos(math.radians(float(a)))

def tangent(a):
    return math.tan(math.radians(float(a)))

def arcsine(a):
    return math.degrees(math.asin(float(a)))

def arccosine(a):
    return math.degrees(math.acos(float(a)))

def arctangent(a):
    return math.degrees(math.atan(float(a)))

def log_base10(a):
    return math.log10(float(a))

def natural_log(a):
    return math.log(float(a))

def exponential(a):
    return math.exp(float(a))

def absolute_value(a):
    return abs(float(a))

def power_of_10(a):
    return 10 ** float(a)

def degrees_to_radians(a):
    return math.radians(float(a))

def radians_to_degrees(a):
    return math.degrees(float(a))

def sinh(a):
    return math.sinh(float(a))

def cosh(a):
    return math.cosh(float(a))

def tanh(a):
    return math.tanh(float(a))

def asinh(a):
    return math.asinh(float(a))

def acosh(a):
    return math.acosh(float(a))

def atanh(a):
    return math.atanh(float(a))

def nth_root(a, n):
    return Decimal(a) ** (1 / Decimal(n))

def gcd(a, b):
    return math.gcd(int(a), int(b))

def lcm(a, b):
    return abs(int(a) * int(b)) // math.gcd(int(a), int(b))

def complex_operations(a, b):
    a = complex(a)
    b = complex(b)
    return (a + b, a - b, a * b, a / b)

def permutations(n, r):
    return math.perm(n, r)

def combinations(n, r):
    return math.comb(n, r)

def random_number(a, b):
    return random.uniform(a, b)

def matrix_multiplication(matrix1, matrix2):
    return np.matmul(matrix1, matrix2)

def eigenvalues_eigenvectors(matrix):
    return linalg.eig(matrix)

def solve_linear_equations(A, B):
    return linalg.solve(A, B)

def numerical_integration(func, a, b):
    return QUADRIC(func, a, b)[0]

def numerical_differentiation(func, x):
    return Derivative(func, x)

def statistical_functions(data):
    return (np.mean(data), np.median(data), np.std(data))

class CalculatorApp(App):
    def build(self):
        self.operators = {
            'Add': add,
            'Subtract': subtract,
            'Multiply': multiply,
            'Divide': divide,
            'Modulus': modulus,
            'Factorial': factorial,
            'Exponentiation': exponentiation,
            'Square Root': square_root,
            'Sine': sine,
            'Cosine': cosine,
            'Tangent': tangent,
            'Arcsine': arcsine,
            'Arccosine': arccosine,
            'Arctangent': arctangent,
            'Logarithm': log_base10,
            'Natural Logarithm': natural_log,
            'Exponential': exponential,
            'Absolute Value': absolute_value,
            'Power of 10': power_of_10,
            'Degrees to Radians': degrees_to_radians,
            'Radians to Degrees': radians_to_degrees,
            'Hyperbolic Sine': sinh,
            'Hyperbolic Cosine': cosh,
            'Hyperbolic Tangent': tanh,
            'Inverse Hyperbolic Sine': asinh,
            'Inverse Hyperbolic Cosine': acosh,
            'Inverse Hyperbolic Tangent': atanh,
            'Nth Root': nth_root,
            'GCD': gcd,
            'LCM': lcm,
            'Complex Operations': complex_operations,
            'Permutations': permutations,
            'Combinations': combinations,
            'Random Number': random_number,
            'Matrix Multiplication': matrix_multiplication,
            'Eigenvalues and Eigenvectors': eigenvalues_eigenvectors,
            'Solve Linear Equations': solve_linear_equations,
            'Numerical Integration': numerical_integration,
            'Numerical Differentiation': numerical_differentiation,
            'Statistical Functions': statistical_functions
        }
        
        layout = BoxLayout(orientation='vertical')

        self.num1 = TextInput(hint_text='Enter the first number you want to insert', multiline=False, input_filter='float')
        self.num2 = TextInput(hint_text='Enter the second number you want to insert', multiline=False, input_filter='float')
        self.operation = TextInput(hint_text='Add,Subtract,Multiply,Divide,Modulus,Factorial,Exponentiation,Square Root,Sine,Cosine,Tangent,Arcsine,Arccosine,Arctangent,Logarithm,Natural Logarithm,Exponential,Absolute Value,Power of 10,Degrees to Radians,Radians to Degrees,Hyperbolic Sine,Hyperbolic Cosine,Hyperbolic Tangent,Inverse Hyperbolic Sine,Inverse Hyperbolic Cosine,Inverse Hyperbolic Tangent,Nth Root,GCD,LCM,Complex Operations,Permutations,Combinations,Random Number,Matrix Multiplication,Eigenvalues and Eigenvectors,Solve Linear Equations,Numerical Integration,Numerical Differentiation,Statistical Functions', multiline=True)
        self.result = Label(text='Result will be displayed here')

        calculate_button = Button(text='Calculate')
        calculate_button.bind(on_press=self.calculate)

        layout.add_widget(self.num1)
        layout.add_widget(self.num2)
        layout.add_widget(self.operation)
        layout.add_widget(calculate_button)
        layout.add_widget(self.result)

        return layout

    def calculate(self, instance):
        try:
            num1 = self.num1.text
            num2 = self.num2.text
            operation = self.operation.text
            if operation in self.operators:
                if operation in ['Matrix Multiplication', 'Eigenvalues and Eigenvectors', 'Solve Linear Equations']:
                    matrix1 = np.array(eval(num1))
                    matrix2 = np.array(eval(num2))
                    result = self.operators[operation](matrix1, matrix2)
                elif operation in ['Numerical Integration', 'Numerical Differentiation']:
                    result = self.operators[operation](eval(num1), eval(num2))
                else:
                    result = self.operators[operation](num1, num2)
                
                formatted_result = format_result(result)
                self.result.text = f"Result: {formatted_result}"
            else:
                self.result.text = "Invalid Operation"
        except Exception as e:
            self.result.text = f"Error: {str(e)}"

if __name__ == "__main__":
    CalculatorApp().run()
