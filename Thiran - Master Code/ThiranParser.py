##########################################
# Parser for Thiran Programming Language
##########################################

import error
from sys import exit

STRUCTURE = ['IF', 'FOR', 'WHILE', 'FUNC']
IO = ['PRINT', 'INPUT']
JUMP = ['RETURN', 'CONTINUE', 'BREAK']

#############################################
# Classes to create different object types
#############################################

##### The parser will create different objects of these classes based on the grammar ####

# number class for number token
class NumberNode:
    def __init__(self, type, value):
        self.type = type
        self.value = value
    def __repr__(self):
        return f'{self.value}'

# stirng class for string token
class StringNode:
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return f'"{self.value}"'

# list class for list token
class ListNode:
    def __init__(self, elements):
        self.elements = elements
    def __repr__(self):
        return f'{self.elements}'

# list index class for implementing list indexing
class ListIndexNode:
    def __init__(self, listelement, index):
        self.listelement = listelement
        self.index = index
    def __repr__(self):
        return f'{self.listelement}[{self.index}]'

# print class for print token type
class PrintNode:
    def __init__(self, expr):
        self.expr = expr
    def __repr__(self):
        return f'Print({self.expr})'

# input class for input token type
class InputNode:
    def __init__(self, expr):
        self.expr = expr
    def __repr__(self):
        return f'Input()'

# binop class for any binary operation
class BinOpNode:
    def __init__(self, leftOperand, op, rightOperand):
        self.leftOp = leftOperand
        self.op = op
        self.rightOp = rightOperand
    def __repr__(self):
        return f'({self.leftOp} {self.op} {self.rightOp})'

# Compop class for any comparison operation
class CompOpNode:
    def __init__(self, leftOperand, op, rightOperand):
        self.leftOp = leftOperand
        self.op = op
        self.rightOp = rightOperand
    def __repr__(self):
        return f'({self.leftOp} {self.op} {self.rightOp})'

# Logicalop class for any logical operation
class LogicalOpNode:
    def __init__(self, leftOperand, op, rightOperand):
        self.leftOp = leftOperand
        self.op = op
        self.rightOp = rightOperand
    def __repr__(self):
        return f'({self.leftOp} {self.op} {self.rightOp})'

# Conditional class for any conditional type token (if elif else)
class ConditionalNode:
    def __init__(self, ifcases, elsecase):
        self.ifcases = ifcases
        self.elsecase = elsecase
    def __repr__(self):
        return f'ifcases:({self.ifcases}) else:({self.elsecase})'

# Function class for any function declaration
class FunctionNode:
    def __init__(self, function_name, formal_params, funcStatements):
        self.function_name = function_name
        self.formal_params = formal_params
        self.funcStatements = funcStatements
    def __repr__(self):
        return f'func {self.function_name}({self.formal_params}): {self.funcStatements}'

# Functioncall class for any function calls
class Func_Call_Statement:
    def __init__(self, func_name, actual_params):
        self.func_name = func_name
        self.actual_params = actual_params
    def __repr__(self):
        return f'func_call {self.func_name}({self.actual_params})'

# import class for any import tokens
class ImportNode:
    def __init__(self, modules_to_import):
        self.modules_to_import = modules_to_import
    def __repr__(self):
        return f'import {self.modules_to_import}'

# return class for any returns from functions
class ReturnNode:
    def __init__(self, returnVal=None):
        self.returnVal = returnVal
    def __repr__(self):
        return f'return {self.returnVal}'

# continue class for using continue inside loops
class ContinueNode:
    def __repr__(self):
        return f'continue'

# continue class for using break inside loops
class BreakNode:
    def __repr__(self):
        return f'break'

# 'for' class for 'for' token (loops)
class ForNode:
    def __init__(self, times_to_loop, repeatStatements):
        self.times_to_loop = times_to_loop
        self.repeatStatements = repeatStatements
    def __repr__(self):
        return f'repeat({self.times_to_loop}): {self.repeatStatements}'

# 'while' class for 'while' token (loops)
class WhileNode:
    def __init__(self, condition, repeatStatements):
        self.condition = condition
        self.repeatStatements = repeatStatements
    def __repr__(self):
        return f'while({self.condition}): {self.repeatStatements}'

# unaryoperator class for unaryoperator types
class UnaryOpNode:
    def __init__(self, op, exp):
        self.op = op
        self.exp = exp
    def __repr__(self):
        return f'{self.op}{self.exp}'

# statement class for each statement
class Statement:
    def __init__(self):
        self.children = []
    def isEmpty(self):
        if len(self.children) == 0:
            return True
        else:
            return False
    def __repr__(self):
        return f'{self.children}'

# assignment class for each assignment statement
class Assignment:
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right
    def __repr__(self):
        return f'{self.left} = {self.right}'

# variable class for each variable token
class Variable:
    def __init__(self, token):
        self.name = token.type
        self.value = token.value
    def __repr__(self):
        return f'{self.name}:{self.value}'


########################################################
# Parser class to perform parsing based on the grammar
########################################################
#### Type of parser implemented - Recursive Descent Parser ####

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = -1
        self.currentToken = None
        self.hasErrored = False

    # to proceed one token forward
    def nextToken(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.currentToken = self.tokens[self.pos]
        else:
            self.currentToken = 'EOF'
    
    # to go back to previous token
    def prevToken(self):
        self.pos -= 1
        if self.pos < len(self.tokens):
            self.currentToken = self.tokens[self.pos]
        else:
            self.currentToken = 'EOF'

    # method to initiate parsing
    def parse(self):
        if len(self.tokens) == 0:
            return [], self.hasErrored
        self.nextToken()
        ast = self.program()
        #print(ast)
        return ast, self.hasErrored

    # grammar rule: program -> [statement]*
    def program(self):
        statements = []
        statements.append(self.statement())
        while self.currentToken != 'EOF':
            statements.append(self.statement())
        prog = Statement()
        for statement in statements:
            prog.children.append(statement)
        return prog
    
    # grammar rule: statement -> assign_statement | function_call_statement
    #              | IO_statement | structure | jump_statement | import_statement
    def statement(self):
        if self.currentToken.type == 'ID':
            self.nextToken()
            if self.currentToken.type == 'ASSIGN':
                self.prevToken()
                res = self.assign_statement()
            else:
                self.prevToken()
                res = self.func_call_statement()
        elif self.currentToken.type in STRUCTURE:
            res = self.structure()
        elif self.currentToken.type in IO:
            res = self.io_statement()
        elif self.currentToken.type in JUMP:
            res = self.jump_statement()
        elif self.currentToken.type == 'IMPORT':
            res = self.import_statement()
        else:
            print("தொகுக்க முடியவில்லை! (codeல் பிழை உள்ளதா என சரிபார்க்கவும்)")
            #print(self.currentToken.type)
            exit()
        return res

    # grammar rule: jump_statement -> RETURN (logical)? | CONTINUE | BREAK
    def jump_statement(self):
        if self.currentToken.type == 'RETURN':
            self.nextToken()
            res = self.logical()
            if res != None:
                res = ReturnNode(res)
            else:
                res = ReturnNode()
        elif self.currentToken.type == 'CONTINUE':
            res = ContinueNode()
            self.nextToken()
        elif self.currentToken.type == 'BREAK':
            res = BreakNode()
            self.nextToken()
        return res

    # grammar rule: IO_statement -> PRINT [LPAREN] (logical
    #               | function_call_statement [[COMMA]([logical]
    #               | [function_call_statement])]*) [RPAREN] | assign_statement
    def io_statement(self):
        if self.currentToken.type == 'PRINT':
            self.nextToken()
            if self.currentToken.type == 'LPAREN':
                self.nextToken()
            else:
                error.Error(u"'காட்டு' பின் '(' எதிர்பார்க்கப்பட்டது")
                self.hasErrored = True
            printList = []
            if self.currentToken.type == 'ID':
                self.nextToken()
                if self.currentToken.type != 'LPAREN':
                    self.prevToken()
                    printList.append(self.logical())
                else:
                    self.prevToken()
                    printList.append(self.func_call_statement())
            else:
                printList.append(self.logical())
            while self.currentToken.type == 'COMMA':
                self.nextToken()
                if self.currentToken.type == 'ID':
                    self.nextToken()
                    if self.currentToken.type != 'LPAREN':
                        self.prevToken()
                        printList.append(self.logical())
                    else:
                        self.prevToken()
                        printList.append(self.func_call_statement())
                else:
                    printList.append(self.logical())
            self.nextToken()
            return PrintNode(printList)
        elif self.currentToken.type == 'INPUT':
            self.nextToken()
            if self.currentToken.type != 'LPAREN':
                error.Error('Syntax பிழை', 'வாங்கு()')
                self.hasErrored = True
            self.nextToken()
            val = self.logical()
            if self.currentToken.type != 'RPAREN':
                error.Error('Syntax பிழை', 'வாங்கு()')
                self.hasErrored = True
            self.nextToken()
            return InputNode(val)

    # grammar rule: assign_statement -> variable ASSIGN [logical | userInput | function_call_statement]
    def assign_statement(self):
        left = self.variable()
        op = self.currentToken
        self.nextToken()
        if self.currentToken.type == 'INPUT':
            right = self.io_statement()
        elif self.currentToken.type == 'ID':
            self.nextToken()
            if self.currentToken.type != 'LPAREN':
                self.prevToken()
                right = self.logical()
            else:
                self.prevToken()
                right = self.logical()
        else:
            right = self.logical()
        res = Assignment(left, op, right)
        return res

    # grammar rule: function_call_statement -> variable [LPAREN] (param [[COMMA][param]]*)* [RPAREN]
    def func_call_statement(self):
        func_name = self.currentToken.value
        self.nextToken()
        if self.currentToken.type != 'LPAREN':
            error.Error("ஒரு செயல்பாகம் அழைப்பிற்கு பின் '(' எதிர்பார்க்கப்பட்டது")
            self.hasErrored = True
            exit()
        else:
            self.nextToken()
            actual_params = []
            while self.currentToken != 'EOF' and self.currentToken.type != 'RPAREN':
                actual_params.append(self.param())
                if self.currentToken.type == 'COMMA':
                    self.nextToken()
                    if self.currentToken.type == 'RPAREN' or self.currentToken.type == 'COMMA':
                        error.Error("செயல்பாகம் அழைப்பில் ',' பின் ஒரு parameter எதிர்பார்க்கப்பட்டது")
                        self.hasErrored = True
                        exit()
            if self.currentToken == 'EOF':
                error.Error("ஒரு செயல்பாகம் அழைப்பிற்கு பின் '()' எதிர்பார்க்கப்பட்டது")
                self.hasErrored = True
                exit()
            if self.currentToken.type == 'RPAREN':
                self.nextToken()
                res = Func_Call_Statement(func_name, actual_params)
            else:
                error.Error("ஒரு செயல்பாகம் அழைப்பிற்கு பின் '()' எதிர்பார்க்கப்பட்டது")
                self.hasErrored = True
                exit()
        return res

    # grammar rule: import_statement -> IMPORT [LPAREN][modulename][[COMMA][modulename]]*[RPAREN]
    def import_statement(self):
        self.nextToken()
        modules_to_import = []
        if self.currentToken == 'EOF':
            error.Error("'இணை'க்கு பின் ஒரு module பெயர் எதிர்பார்க்கப்பட்டது")
            self.hasErrored = True
            exit()
        if self.currentToken.type == 'LSQBR':
            self.nextToken()
        else:
            error.Error("'இணை'க்கு பின் '[' எதிர்பார்க்கப்பட்டது")

            self.hasErrored = True
            exit()
        while self.currentToken != 'EOF' and self.currentToken.type != 'RSQBR':
            if self.currentToken.type == 'ID':
                modules_to_import.append(self.currentToken.value)
                self.nextToken()
            else:
                error.Error("module பெயரில் பிழை உள்ளது")
                self.hasErrored = True
                exit()
            if self.currentToken.type == 'COMMA':
                self.nextToken()
                if self.currentToken.type == 'COMMA' or self.currentToken.type == 'RSQBR':
                    error.Error("',' பின் ஒரு module பெயர் எதிர்பார்க்கப்பட்டது", "இணை")
                    self.hasErrored = True
                    exit()
        if self.currentToken == 'EOF':
            error.Error("'இணை'க்கு பின் '[]' எதிர்பார்க்கப்பட்டது")
            self.hasErrored = True
            exit()
        if self.currentToken.type == 'COMMA':
            error.Error("',' தவறான பயன்பாடு",  "இணை")
            self.hasErrored = True
            exit()
        if self.currentToken.type == 'RSQBR':
            self.nextToken()
        if len(modules_to_import) == 0:
            error.Error("ஒரு module பெயராவது எதிர்பார்க்கப்பட்டது", "இணை")
            self.hasErrored = True
            exit()
        res = ImportNode(modules_to_import)
        return res

    # grammar rule: structure -> conditional | loop | function
    def structure(self):
        if self.currentToken.type == 'IF':
            res = self.conditional()
            return res
        # grammar rule: loop -> for_loop | while_loop (avoided coding a method for loop)
        elif self.currentToken.type == 'FOR':
            res = self.for_loop()
            return res
        elif self.currentToken.type == 'WHILE':
            res = self.while_loop()
            return res
        elif self.currentToken.type == 'FUNC':
            res = self.function()
            return res
        
    # grammar rule: function -> func variable [LPAREN] (variable [[COMMA][variable]]*)* [RPAREN] [:] (statement)* END 
    def function(self):
        self.nextToken()
        if self.currentToken.type != 'ID':
            error.Error("'செயல்பாகம்' பின் செயல்பாகத்திற்கான பெயர் எதிர்பார்க்கப்பட்டது")
            self.hasErrored = True
            exit()
        else:
            function_name = self.currentToken.value
            self.nextToken()
            if self.currentToken.type != 'LPAREN':
                error.Error("'செயல்பாகம்' பின் '()' எதிர்பார்க்கப்பட்டது")
                self.hasErrored = True
                exit()
            else:
                self.nextToken()
                formal_params = []
                while self.currentToken != 'EOF' and self.currentToken.type != 'RPAREN' and self.currentToken.type != 'COLON':
                    formal_params.append(self.variable())
                    if self.currentToken.type == 'COMMA':
                        self.nextToken()
                        if self.currentToken.type == 'RPAREN' or self.currentToken.type == 'COMMA':
                            error.Error("செயல்பாகத்தில் ',' பின் ஒரு parameter எதிர்பார்க்கப்பட்டது")
                            self.hasErrored = True
                            exit()
                if self.currentToken == 'EOF':
                    error.Error("'செயல்பாகம்' பின் '()' எதிர்பார்க்கப்பட்டது")
                    self.hasErrored = True
                    exit()
                if self.currentToken.type == 'RPAREN':
                    self.nextToken()
                    if self.currentToken.type != 'COLON':
                        error.Error("'செயல்பாகம்' பின் ':' எதிர்பார்க்கப்பட்டது")
                        self.hasErrored = True
                        exit()
                    else:
                        self.nextToken()
                        funcStatements = []
                        while self.currentToken != 'EOF' and self.currentToken.type != 'END':
                            funcStatements.append(self.statement())
                        if self.currentToken == 'EOF':
                            error.Error("'செயல்பாகம்()' முடிவில் 'முடி' எதிர்பார்க்கப்பட்டது")
                            self.hasErrored = True
                            exit()
                        elif self.currentToken.type == 'END':
                            self.nextToken()
                            res = FunctionNode(function_name, formal_params, funcStatements)
                            return res
                        else:
                            error.Error("'செயல்பாகம்()' முடிவில் 'முடி' எதிர்பார்க்கப்பட்டது")
                            self.hasErrored = True
                            exit()
                else:
                    error.Error("'செயல்பாகம்' பின் '()' எதிர்பார்க்கப்பட்டது")
                    self.hasErrored = True
                    exit()

    # grammar rule: for_loop -> REPEAT [LPAREN] (int|variable) [RPAREN] [:] (statement)* END
    def for_loop(self):
        self.nextToken()
        if self.currentToken.type != 'LPAREN':
            error.Error("'திரும்பச்செய்' பின் '()' எதிர்பார்க்கப்பட்டது")
            self.hasErrored = True
            exit()
        else:
            self.nextToken()
            times_to_loop = self.logical()
            try:
                if times_to_loop.name == 'RPAREN':
                    error.Error("'திரும்பச்செய்()' உள் முழுஎண் எதிர்பார்க்கப்பட்டது")
                    self.hasErrored = True 
                    exit()
            except:
                pass
            if self.hasErrored:
                exit()
            if self.currentToken.type != 'RPAREN':
                error.Error("'திரும்பச்செய்' பின் '()' எதிர்பார்க்கப்பட்டது")
                self.hasErrored = True
                exit()
            else:
                self.nextToken()
            if self.currentToken.type != 'COLON':
                error.Error("'திரும்பச்செய்()' பின் ':' எதிர்பார்க்கப்பட்டது")
                self.hasErrored = True
                exit()
            else:
                self.nextToken()
                repeatStatements = []
                while self.currentToken != 'EOF' and self.currentToken.type != 'END':
                    repeatStatements.append(self.statement())
                if self.currentToken == 'EOF':
                    error.Error("'திரும்பச்செய்()' முடிவில் 'முடி' எதிர்பார்க்கப்பட்டது")
                    self.hasErrored = True
                    exit()
                elif self.currentToken.type == 'END':
                    self.nextToken()
                    res = ForNode(times_to_loop, repeatStatements)
                    return res
                else:
                    error.Error("'திரும்பச்செய்()' முடிவில் 'முடி' எதிர்பார்க்கப்பட்டது")
                    self.hasErrored = True
                    exit()

    # grammar rule: while_loop -> WHILE [LPAREN] logical [RPAREN] [:] (statement)* END
    def while_loop(self):
        self.nextToken()
        if self.currentToken.type != 'LPAREN':
            error.Error("'உண்மைவரை' பின் '()' எதிர்பார்க்கப்பட்டது")
            self.hasErrored = True
            exit()
        else:
            self.nextToken()
            condition = self.logical()
            try:
                if condition.name == 'RPAREN':
                    error.Error("'உண்மைவரை( )' உள் ஒரு Logical Condition எதிர்பார்க்கப்பட்டது")
                    self.hasErrored = True 
                    exit()
            except:
                pass
            if self.hasErrored:
                exit()
            if self.currentToken.type != 'RPAREN':
                error.Error("'உண்மைவரை' பின் '()' எதிர்பார்க்கப்பட்டது")
                self.hasErrored = True
                exit()
            else:
                self.nextToken()
            if self.currentToken.type != 'COLON':
                error.Error("'உண்மைவரை()' பின் ':' எதிர்பார்க்கப்பட்டது")
                self.hasErrored = True
                exit()
            else:
                self.nextToken()
                repeatStatements = []
                while self.currentToken != 'EOF' and self.currentToken.type != 'END':
                    repeatStatements.append(self.statement())
                if self.currentToken == 'EOF':
                    error.Error("'உண்மைவரை()' முடிவில் 'முடி' எதிர்பார்க்கப்பட்டது")
                    self.hasErrored = True
                    exit()
                elif self.currentToken.type == 'END':
                    self.nextToken()
                    res = WhileNode(condition, repeatStatements)
                    return res
                else:
                    error.Error("'உண்மைவரை()' முடிவில் 'முடி' எதிர்பார்க்கப்பட்டது")
                    self.hasErrored = True
                    exit()

    # grammar rule: conditional -> IF [LPAREN] (logical) [RPAREN] [:] 
    #           statement ([ELIF][LPAREN] (logical) [RPAREN] [:] statement)* 
    #           ([ELSE] [LPAREN] (logical) [RPAREN] [:] statement)? END
    def conditional(self):
        self.nextToken()
        if self.currentToken.type != 'LPAREN':
            error.Error("'சரியெனில்' பின் '()' எதிர்பார்க்கப்பட்டது")
            self.hasErrored = True
            exit()
        condition = self.logical()
        ifcases = []
        if self.currentToken.type == 'COLON':
            self.nextToken()
            ifStatements = []
            while self.currentToken != 'EOF' and self.currentToken.type != 'ELSE' and self.currentToken.type != 'ELIF' and self.currentToken.type != 'END':
                ifStatements.append(self.statement())
            ifcases.append((condition, ifStatements))
        else:
            error.Error("'சரியெனில்()' பின் ':' எதிர்பார்க்கப்பட்டது")
            print(self.currentToken)
            self.nextToken()
            print(self.currentToken)
            self.nextToken()
            print(self.currentToken)
            self.nextToken()
            print(self.currentToken)
            self.hasErrored = True
            exit()
        if self.currentToken == 'EOF':
            error.Error("Conditional Block முடிவில் 'முடி' எதிர்பார்க்கப்பட்டது")
            self.hasErrored = True
            exit()
        if self.currentToken.type == 'END':
            self.nextToken()
            res = ConditionalNode(ifcases, None)
            return res
        else:
            while self.currentToken.type == 'ELIF' and self.currentToken != 'EOF':
                self.nextToken()
                condition = self.logical()
                if self.currentToken.type == 'COLON':
                    self.nextToken()
                    elifStatements = []
                    while self.currentToken.type != 'ELIF' and self.currentToken.type != 'ELSE' and self.currentToken.type != 'END' and self.currentToken != 'EOF':
                        elifStatements.append(self.statement())
                    ifcases.append((condition, elifStatements))
                else:
                    error.Error("'அல்லதெனில்()' பின் ':' எதிர்பார்க்கப்பட்டது")
            if self.currentToken == 'EOF':
                error.Error("Conditional Block முடிவில் 'முடி' எதிர்பார்க்கப்பட்டது")
                self.hasErrored = True
                exit()
            if self.currentToken.type == 'END':
                self.nextToken()
                res = ConditionalNode(ifcases, None)
                return res
            else:
                if self.currentToken.type == 'ELSE' and self.currentToken != 'EOF':
                    self.nextToken()
                    if self.currentToken.type == 'COLON':
                        self.nextToken()
                        elseStatements = []
                        while self.currentToken != 'EOF' and self.currentToken.type != 'END':
                            elseStatements.append(self.statement())
                    else:
                        error.Error("'தவறெனில்' பின் ':' எதிர்பார்க்கப்பட்டது")
                    if self.currentToken == 'EOF':
                        error.Error("Conditional Block முடிவில் 'முடி' எதிர்பார்க்கப்பட்டது")
                        self.hasErrored = True
                        exit()
                    elif self.currentToken.type == 'END':
                        self.nextToken()
                        res = ConditionalNode(ifcases, elseStatements)
                        return res
                    else:
                        error.Error("Conditional Block முடிவில் 'முடி' எதிர்பார்க்கப்பட்டது")
                        self.hasErrored = True
                        exit()

    # grammar rule: variable -> ID
    def variable(self):
        res = Variable(self.currentToken)
        self.nextToken()
        return res

    # grammar rule: logical -> comparison ([AND|OR] comparison)*
    def logical(self):
        leftComp = self.comparison()
        while self.currentToken != 'EOF' and self.currentToken.type in ('AND', 'OR'):
            operator = self.currentToken.type
            self.nextToken()
            rightComp = self.comparison()
            leftComp = LogicalOpNode(leftComp, operator, rightComp)
        return leftComp

    # grammar rule: comparison -> [NOT] comparison | expression ([==|!=|<|>|<=|>=] expression)*
    def comparison(self):
        if self.currentToken != 'EOF' and self.currentToken.type == 'NOT':
            self.nextToken()
            temp = self.comparison()
            node = UnaryOpNode('NOT', temp)
            return node
        leftExp = self.expression()
        while self.currentToken != 'EOF' and self.currentToken.type in ('EQ', 'NOT_EQ', 'LESS_EQ', 'LESS', 'GREATER_EQ', 'GREATER'):
            operator = self.currentToken.type
            self.nextToken()
            rightExp = self.expression()
            leftExp = CompOpNode(leftExp, operator, rightExp)
        return leftExp

    # grammar rule: expression -> term [[+ | -] term]*
    def expression(self):
        leftTerm = self.term()
        while self.currentToken != 'EOF' and self.currentToken.type in ('PLUS', 'MINUS'):
            operator = self.currentToken.type
            self.nextToken()
            rightTerm = self.term()
            leftTerm = BinOpNode(leftTerm, operator, rightTerm)
        return leftTerm

    # grammar rule: term -> factor [[* | / | %] factor]*
    def term(self):
        leftFactor = self.factor()
        while self.currentToken != 'EOF' and self.currentToken.type in ('MUL', 'DIV', 'MOD'):
            operator = self.currentToken.type
            self.nextToken()
            rightFactor = self.factor()
            leftFactor = BinOpNode(leftFactor, operator, rightFactor)
        return leftFactor

    # grammar rule: factor -> [plus | minus] factor | [int | float] 
    #       | [LPAREN]logical[RPAREN] | variable | string 
    #       | list | listIndex | function_call_statement
    def factor(self):
        if self.currentToken != 'EOF':
            if (self.currentToken.type == 'PLUS' or self.currentToken.type == 'MINUS'):
                unaryOp = self.currentToken.type
                self.nextToken()
                factor = UnaryOpNode(unaryOp, self.factor())
            elif self.currentToken.type in ('INT', 'FLOAT'):
                factor = NumberNode(self.currentToken.type, self.currentToken.value)
                self.nextToken()
            elif self.currentToken.type == 'LPAREN':
                self.nextToken()
                factor = self.logical()
                self.nextToken()
            elif self.currentToken.type == 'STRING':
                factor = StringNode(self.currentToken.value)
                self.nextToken()
            elif self.currentToken.type == 'TRUE':
                factor = NumberNode('INT', 1)
                self.nextToken()
            elif self.currentToken.type == 'FALSE':
                factor = NumberNode('INT', 0)
                self.nextToken()
            elif self.currentToken.type == 'ID':
                self.nextToken()
                if self.currentToken.type == 'LPAREN':
                    self.prevToken()
                    factor = self.func_call_statement()
                else:
                    self.prevToken()
                    factor = self.variable()
                    if self.currentToken.type == 'LSQBR':
                        index = self.listIndex()
                        factor = ListIndexNode(factor, index)
            elif self.currentToken.type == 'LSQBR':
                elements = self.handleList()
                factor = ListNode(elements)
                if self.currentToken != 'EOF' and self.currentToken.type == 'LSQBR':
                    index = self.listIndex()
                    factor = ListIndexNode(factor, index)
            else:
                return 'None'
            return factor
        else:
            return None

    # grammar rule: list -> [LSQBR] [(factor)[[COMMA][factor]]*]* [RSQBR]
    def handleList(self): # name changed to handleList to avoid conflict with python's default list() method
        self.nextToken()
        if self.currentToken != 'EOF':
            elements = []
            while self.currentToken.type != 'RSQBR':
                elements.append(self.factor())
                if self.currentToken.type == 'COMMA':
                    self.nextToken()
                    if self.currentToken.type == 'RSQBR' or self.currentToken.type == 'COMMA':
                        error.Error("',' பின் ஒரு element எதிர்பார்க்கப்பட்டது", "பட்டியல்")
                        self.hasErrored = True
                        exit()
            self.nextToken()
            return elements
        else:
            return None

    # grammar rule: listIndex -> [list | variable][LSQBR][int|variable|logical][RSQBR]
    def listIndex(self):
        self.nextToken()
        try:
            index = self.logical()
            if self.currentToken.type != 'RSQBR':
                    error.Error("'பட்டியல்[ ' பின் ']' எதிர்பார்க்கப்பட்டது")
                    self.hasErrored = True
                    exit()
            else:
                self.nextToken()
        except:
        #else:
            error.Error("'பட்டியல்[ ]' உள் ஒரு முழுஎண்(0 அல்லது அதற்கு மேல்) எதிர்பார்க்கப்பட்டது")
            self.hasErrored = True
            exit()
        '''
        if self.currentToken.type == 'INT':
            index = NumberNode(self.currentToken.type, self.currentToken.value)
            self.nextToken()
            if self.currentToken.type != 'RSQBR':
                error.Error("'பட்டியல்[ ' பின் ']' எதிர்பார்க்கப்பட்டது")
                self.hasErrored = True
                exit()
            else:
                self.nextToken()
        elif self.currentToken.type == 'ID':
            index = Variable(self.currentToken)
            self.nextToken()
            if self.currentToken.type != 'RSQBR':
                error.Error("'பட்டியல்[ ' பின் ']' எதிர்பார்க்கப்பட்டது")
                self.hasErrored = True
                exit()
            else:
                self.nextToken()
        '''
        
        return index

    # grammar rule: param -> variable | [int | float] 
    #              | string | [plus | minus] param | list | listIndex | logical
    def param(self):
        if self.currentToken != 'EOF':
            if (self.currentToken.type == 'PLUS' or self.currentToken.type == 'MINUS'):
                unaryOp = self.currentToken.type
                self.nextToken()
                factor = UnaryOpNode(unaryOp, self.param())
            elif self.currentToken.type in ('INT', 'FLOAT'):
                factor = NumberNode(self.currentToken.type, self.currentToken.value)
                self.nextToken()
            elif self.currentToken.type == 'STRING':
                factor = StringNode(self.currentToken.value)
                self.nextToken()
            elif self.currentToken.type == 'TRUE':
                factor = NumberNode('INT', 1)
                self.nextToken()
            elif self.currentToken.type == 'FALSE':
                factor = NumberNode('INT', 0)
                self.nextToken()
            elif self.currentToken.type == 'ID':
                #factor = self.variable()
                factor = self.logical()
                if self.currentToken.type == 'LSQBR':
                    index = self.listIndex()
                    factor = ListIndexNode(factor, index)
            elif self.currentToken.type == 'LSQBR':
                elements = self.handleList()
                factor = ListNode(elements)
                if self.currentToken.type == 'LSQBR':
                    index = self.listIndex()
                    factor = ListIndexNode(factor, index)
            else:
                return None
            return factor
        else:
            return None


#########################################################
# (c) 2023 Surya Narayanan <suryanarayanansg@gmail.com>
#########################################################
#       Grammar for Thiran Programming Language
#########################################################

# Some modifications in grammar were done in code either for simplicity or for added functionality

'''
program -> [statement]*
statement -> assign_statement | function_call_statement | IO_statement | structure | jump_statement | import_statement
structure -> conditional | loop | function
function -> func variable [LPAREN] (variable [[COMMA][variable]]*)* [RPAREN] [:] (statement)* END
loop -> for_loop | while_loop
for_loop -> REPEAT [LPAREN] (int|variable) [RPAREN] [:] (statement)* END
while_loop -> WHILE [LPAREN] logical [RPAREN] [:] (statement)* END
conditional -> IF [LPAREN] (logical) [RPAREN] [:] statement ([ELIF][LPAREN] (logical) [RPAREN] [:] statement)* ([ELSE] [LPAREN] (logical) [RPAREN] [:] statement)? END
assign_statement -> variable ASSIGN [logical | userInput | function_call_statement]
function_call_statement -> variable [LPAREN] (param [[COMMA][param]]*)* [RPAREN]
IO_statement -> PRINT [LPAREN] (logical | function_call_statement [[COMMA]([logical]|[function_call_statement])]*) [RPAREN] | assign_statement
jump_statement -> RETURN (logical)? | CONTINUE | BREAK
import_statement -> IMPORT [LPAREN][modulename][[COMMA][modulename]]*[RPAREN]
variable -> ID
logical -> comparison ([AND|OR] comparison)*
comparison -> [NOT] comparison | expression ([==|!=|<|>|<=|>=] expression)*
expression -> term [[+ | -] term]*
term -> factor [[* | / | %] factor]*
factor -> [plus | minus] factor | [int | float] | [LPAREN]logical[RPAREN] | variable | string | list | listIndex | function_call_statement
list -> [LSQBR] [(factor)[[COMMA][factor]]*]* [RSQBR]
listIndex -> [list | variable][LSQBR][int|variable|logical][RSQBR]
int -> [digit]*
float -> [digit]* ['.'] [digit]*
digit -> 0|1|2|3|4|5|6|7|8|9
string -> ["][letters|digit|symbols]*["]
param -> variable | [int | float] | string | [plus | minus] param | list | listIndex | logical
'''