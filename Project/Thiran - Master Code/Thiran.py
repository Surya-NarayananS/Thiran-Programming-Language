########################################################
# (c) 2023 Surya Narayanan <suryanarayanansg@gmail.com>
########################################################
##############################################
# Master Code for Thiran Programming Language
##############################################

########################
# Code files for Thiran
########################
# 1. Thiran.py
# 2. ThiranLexer.py
# 3. ThiranParser.py
# 4. ThiranInterpreter.py
# 5. error.py
# 6. globals.py
# 7. ThiranImport.py
# 8. ThiranBuiltIns.py
# ----------------------------

# திறன் நிரலாக்க மொழி (Thiran Programming Language)
# Check Thiran's repo: https://github.com/Surya-NarayananS/Thiran-Programming-Language
# Would be happy if you can contribute to this open project
# வாழ்க தமிழ் !!! நன்றி !

##########################################################

import ThiranLexer, ThiranParser, ThiranInterpreter
import sys
from sys import exit

# function to tokenize the code
def lexer(code):
    lexer = ThiranLexer.Lexer(code)
    tokens, hasError = lexer.tokenize()
    return None if hasError else tokens

# function to parse the output from lexer
def parser(tokens):
    parser = ThiranParser.Parser(tokens)
    ast, hasError = parser.parse()
    return None if hasError else ast

# function to interpret the output from parser
def interpreter(ast):
    interpreter = ThiranInterpreter.Interpreter(ast)
    res, hasError = interpreter.interpret()
    #print('Symbol table:', interpreter.SYMBOL_TABLE)
    return None if hasError else res

try:
    # checks if more than one file is passed to Thiran Compiler
    n = len(sys.argv)
    if n > 2:
        print("தொகுத்து  இயக்க ஏதேனும் ஒரு '.ti' fileயை இணைக்கவும்!")
        exit()
    else:
        filename = sys.argv[1]
        # checks if the passed file has a '.ti' extension to it
        if filename.lower().endswith(".ti"):
            pass
        else:
            exit()
except:
    #filename = "test.ti"
    print("திறன் நிரல் மொழி - தொகுத்து  இயக்க ஏதேனும் '.ti' fileயை இணைக்கவும்!")
    exit()

# fuction to compile and execute the '.ti' file
def compile_run(filename):
    try:
        code_file = open(filename, "r", encoding="utf-8")
        code = code_file.read()
        code_file.close()
    except:
        print("fileயை கண்டுபிடிக்க முடியவில்லை: ", filename)
        exit()
    lex = lexer(code)
    if lex != None:
        #print('lexer:', lex)
        parse = parser(lex)
        #print('parser:', parse)
        if parse != None:
            res = interpreter(parse)

compile_run(filename)

#end = input("\nப்ரோக்ராம் இயக்கம் முடிந்தது. வெளியேற ஏதேனும் கீயை அழுத்தவும்.")