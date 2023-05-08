##############################################
# Code to handle Import statements in Thiran
##############################################

import ThiranLexer, ThiranParser, ThiranInterpreter, error
from sys import exit

def lexer(code):
    lexer = ThiranLexer.Lexer(code)
    tokens, hasError = lexer.tokenize()
    return None if hasError else tokens

def parser(tokens):
    parser = ThiranParser.Parser(tokens)
    ast, hasError = parser.parse()
    return None if hasError else ast

def interpreter(ast):
    interpreter = ThiranInterpreter.Interpreter(ast)
    res, hasError = interpreter.interpret()
    # the module file has been executed
    # After execution, its symbol table and declared functions are returned to be appended with main file
    return None if hasError else [interpreter.SYMBOL_TABLE, interpreter.DECLARED_FUNCTIONS]

def load_module(module_name):
    module_name = module_name + '.ti'
    try:
        code_file = open(module_name, "r", encoding="utf-8")
        code = code_file.read()
        code_file.close()
    except Exception as e:
        error.Error(module_name, "கண்டுபிடிக்க முடியவில்லை (ஒரே pathல் உள்ளதை உறுதிசெய்க)")
        exit()
    lex = lexer(code)
    if lex != None:
        parse = parser(lex)
        if parse != None:
            res = interpreter(parse)
            if res != None:
                return res
            else:
                error.Error(module_name, "இணைக்க முடியவில்லை")
                exit()
        else:
            error.Error(module_name, "இணைக்க முடியவில்லை")
            exit()
    else:
        error.Error(module_name, "இணைக்க முடியவில்லை")
        exit()