import ThiranLexer, ThiranParser, ThiranInterpreter, tamil
import sys

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
    #print('Symbol table:', interpreter.SYMBOL_TABLE)
    return None if hasError else res       
try:
    n = len(sys.argv)
    if n > 2:
        print("Invalid Argument (Expected a single file)")
        exit()
    else:
        filename = sys.argv[1]
except:
    filename = 'test.ti'

code = open(filename, "r", encoding="utf-8").read()
lex_res = lexer(code)
if lex_res != None:
    #print('lexer:', lex_res)
    parse_res = parser(lex_res)
    #print('parser:', parse_res)
    if parse_res != None:
       result = interpreter(parse_res)
