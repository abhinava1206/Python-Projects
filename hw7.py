import numpy as np
import re
from numpy import linalg as LA
import sympy as sp
import math
from fractions import Fraction

#function: finds the correlation coefficient of 2 samples
#@param x: observations of x in the form of a numpy array
#@param y: observations of y in the form of a numpy array
#@return correlation coefficient of x and y
def corrcoeff(x,y):
    deviation_x = x.copy()
    deviation_x = deviation_x.astype(float) # The deviation vector should take decimals
    deviation_y = y.copy()
    deviation_y = deviation_y.astype(float)
    mean_x = np.sum(x)/float(len(x)) # Finds the mean
    mean_y = np.sum(y)/float(len(y))
    for i in range(len(x)):
        deviation_x[i] = x[i] - mean_x # Deviations
        deviation_y[i] = y[i] - mean_y
    return np.dot(deviation_x,deviation_y)/float((LA.norm(deviation_x,2)*LA.norm(deviation_y,2))) # Formula from lecture    
 
#function: finds the probability that the standard normal distribution falls between a and b
#@param a: real number
#@param b: real number
#@return probability        
def normalcurve(a,b):
    x = sp.symbols('x')
    k = 1/float(math.sqrt(2*np.pi)) # normalization constant
    z = float(sp.integrate(sp.exp((-x**2)/2),(x,a,b))) # formula from lecture
    return z*k

#function: helper function to find out elements of a molecule(Compound)
#@param molecule: input the compound
#@return a dictionary of elements(keys) and the corresponding number of atoms(values)

def elements_dictionary(molecule):
    elements_dictionary = {}
    elements = re.findall(r'[A-Z][a-z]?[\d]?', molecule)
    for e in elements:
        if re.search(r'[A-Z][a-z]?$',e):
            elements_dictionary[e] = 1
        else:
            num = int(re.search(r'\d+',e).group(0))
            e = re.sub(r'(\w+)\d+',r'\1',e)
            
            elements_dictionary[e] = num
    return elements_dictionary

#function: takes as input a string (with no spaces) of a chemical equation and returns the string of the balanced equation
#@param: eq iput string of the equation
#@return a string of the balanced equations
def balance(eq):
    list_of_elements = [] 
    compounds = eq.split("=") # splits the equation into its left and right
    lhs = compounds[0] # left hand side
    rhs = compounds[1] # right hand side
    l_compounds = lhs.split("+") # splits the left hand side into compounds
    r_compounds = rhs.split("+") # splits the right hand side into its compounds
    l_list = []
    r_list = []
    for k in range(len(l_compounds)):
        l_list.append(elements_dictionary(l_compounds[k])) 
        molecule = l_compounds[k]
        elements = re.findall(r'[A-Z][a-z]?[\d]?', molecule)
        for e in elements:
            if re.search(r'[A-Z][a-z]?$',e):
                list_of_elements.append(e) # If e is an element
            else:
                e = re.sub(r'(\w+)\d+',r'\1',e) # Removes the number, then adds the element alone to the list
                list_of_elements.append(e)
    list_of_elements = list(set(list_of_elements))  # Removes duplicates            
    for k in range(len(r_compounds)):
        r_list.append(elements_dictionary(r_compounds[k]))
    
    a = int(len(list_of_elements)) # Number of elements
    b = int(len(l_compounds) + len(r_compounds)) # Number of terms
    M = np.zeros((a,b+1)) # Dimensions of the augmented matrix
    for i in range(len(list_of_elements)):
        e = list_of_elements[i]
        for j in range(len(l_list) + len(r_list)):
            if (j<len(l_list)):
                if e in l_list[j].keys():
                    atoms = (l_list[j])[e]
                    M[i][j] = atoms # Sets the corresponding entry of the matrix
            else:
                if e in r_list[j - len(l_list)].keys():
                    atoms = -(r_list[j - len(l_list)])[e]
                    M[i][j] = atoms # Sets the corresponding entry of the matrix
    M = M.astype(int) # Converts the matrix to be of integers
    a,b,c,d,e,f,g,h,i,j,k,l,m,n = sp.symbols("a,b,c,d,e,f,g,h,i,j,k,l,m,n")
    variables = [a,b,c,d,e,f,g,h,i,j,k,m,n]
    solution = sp.solve_linear_system(sp.Matrix(M), a,b,c,d,e,f,g,h,i,j,k,m,n)
    variables_used = []
    for v in variables:
        if v in solution.keys():
            variables_used.append(v) # If the variable has been used, append it to the list
    last_variable = variables[len(variables_used)] # Identifies the last variable used
    coeffs = []
    for v in variables_used:
        coeffs.append(solution[v]/last_variable) 
    coeffs.append(1) # The last coefficient is set to 1 by default
    fractions = []
    for i in range(len(coeffs)-1):
        f = coeffs[i].subs(last_variable,1) # To remove the variables from the coefficients
        fractions.append(Fraction(str(f)).limit_denominator())
    fractions.append(Fraction(1)) # Appends the last 1
        
    denominators = []
    for fraction in fractions:
        denominators.append(fraction.denominator)
    denominators = list(set(denominators)) # Removes duplicates
    product = sp.lcm(denominators) # Finds the LCM
    new_coeffs = []
    for fraction in fractions:
        new_coeffs.append(int(fraction.numerator*product/
                                  fraction.denominator)) # Multiplies by the LCM to remove fractions
    positive_coeffs = []
    for c in new_coeffs:
        if c < 0:
            c = c*-1 # Removes negatives
            positive_coeffs.append(str(c))
        else:
            positive_coeffs.append(str(c))
    if positive_coeffs[0] == "1":
        balanced = l_compounds[0] # Initiates the result string
    else:
        balanced = positive_coeffs[0] + l_compounds[0]
    for i in range(len(l_compounds)):
        if positive_coeffs[i] == "1":
            string = "" # If the coefficient is 1, then add nothing 
        else:
            string = positive_coeffs[i]
        if i!=0:
            balanced += "+" + string + l_compounds[i]
    balanced += "=" # Now for the right side
    for i in range(len(r_compounds)):
        if positive_coeffs[i + len(l_compounds)] == "1":
            string = ""
        else:
            string = positive_coeffs[i + len(l_compounds)]
        if i == 0:
            balanced += string + r_compounds[i]
        else:
            balanced += "+" + string + r_compounds[i]
            
    return balanced
        
