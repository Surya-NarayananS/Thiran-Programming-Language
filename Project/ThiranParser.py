import error

STRUCTURE = ['IF', 'FOR', 'WHILE', 'FUNC']
IO = ['PRINT', 'INPUT']
JUMP = ['RETURN', 'CONTINUE', 'BREAK']

class NumberNode:
    def __init__(self, type, value):
        self.type = type
        self.value = value
    def __repr__(self):
        return f'{self.value}'

class StringNode:
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return f'"{self.value}"'

class PrintNode:
    def __init__(self, expr):
        self.expr = expr
    def __repr__(self):
        return f'Print({self.expr})'

class InputNode:
    def __init__(self, expr):
        self.expr = expr
    def __repr__(self):
        return f'Input()'

class BinOpNode:
    def __init__(self, leftOperand, op, rightOperand):
        self.leftOp = leftOperand
        self.op = op
        self.rightOp = rightOperand
    def __repr__(self):
        return f'({self.leftOp} {self.op} {self.rightOp})'
    
class CompOpNode:
    def __init__(self, leftOperand, op, rightOperand):
        self.leftOp = leftOperand
        self.op = op
        self.rightOp = rightOperand
    def __repr__(self):
        return f'({self.leftOp} {self.op} {self.rightOp})'
    
class LogicalOpNode:
    def __init__(self, leftOperand, op, rightOperand):
        self.leftOp = leftOperand
        self.op = op
        self.rightOp = rightOperand
    def __repr__(self):
        return f'({self.leftOp} {self.op} {self.rightOp})'
    
class ConditionalNode:
    def __init__(self, ifcases, elsecase):
        self.ifcases = ifcases
        self.elsecase = elsecase
    def __repr__(self):
        return f'ifcases:({self.ifcases}) else:({self.elsecase})'

class FunctionNode:
    def __init__(self, function_name, formal_params, funcStatements):
        self.function_name = function_name
        self.formal_params = formal_params
        self.funcStatements = funcStatements
    def __repr__(self):
        return f'func {self.function_name}({self.formal_params}): {self.funcStatements}'

class Func_Call_Statement:
    def __init__(self, func_name, actual_params):
        self.func_name = func_name
        self.actual_params = actual_params
    def __repr__(self):
        return f'func_call {self.func_name}({self.actual_params})'

class ReturnNode:
    def __init__(self, returnVal=None):
        self.returnVal = returnVal
    def __repr__(self):
        return f'return {self.returnVal}'

class ContinueNode:
    def __repr__(self):
        return f'continue'

class BreakNode:
    def __repr__(self):
        return f'break'

class ForNode:
    def __init__(self, times_to_loop, repeatStatements):
        self.times_to_loop = times_to_loop
        self.repeatStatements = repeatStatements
    def __repr__(self):
        return f'repeat({self.times_to_loop}): {self.repeatStatements}'

class WhileNode:
    def __init__(self, condition, repeatStatements):
        self.condition = condition
        self.repeatStatements = repeatStatements
    def __repr__(self):
        return f'while({self.condition}): {self.repeatStatements}'

class UnaryOpNode:
    def __init__(self, op, exp):
        self.op = op
        self.exp = exp
    def __repr__(self):
        return f'{self.op}{self.exp}'

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

class Assignment:
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right
    def __repr__(self):
        return f'{self.left} = {self.right}'

class Variable:
    def __init__(self, token):
        self.name = token.type
        self.value = token.value
    def __repr__(self):
        return f'{self.name}:{self.value}'

# Recursive Descent Parser for to parse the grammar

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = -1
        self.currentToken = None
        self.hasErrored = False

    def nextToken(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.currentToken = self.tokens[self.pos]
        else:
            self.currentToken = 'EOF'
    
    def prevToken(self):
        self.pos -= 1
        if self.pos < len(self.tokens):
            self.currentToken = self.tokens[self.pos]
        else:
            self.currentToken = 'EOF'

    def parse(self):
        if len(self.tokens) == 0:
            return [], self.hasErrored
        self.nextToken()
        ast = self.program()
        #print(ast)
        return ast, self.hasErrored

    def program(self):
        statements = []
        statements.append(self.statement())
        while self.currentToken != 'EOF':
            statements.append(self.statement())
        prog = Statement()
        for statement in statements:
            prog.children.append(statement)
        return prog

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
        else:
            print("Grammar doesn't support! (Please check the syntax of the code)")
            exit()
        return res

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

    def io_statement(self):
        if self.currentToken.type == 'PRINT':
            self.nextToken()
            if self.currentToken.type == 'LPAREN':
                self.nextToken()
            else:
                error.Error(u"Expected '(' after காட்டு")
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
                error.Error('Invalid Syntax', 'INPUT')
                self.hasErrored = True
            self.nextToken()
            val = self.logical()
            if self.currentToken.type != 'RPAREN':
                error.Error('Invalid Syntax', 'INPUT')
                self.hasErrored = True
            self.nextToken()
            return InputNode(val)

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
                right = self.func_call_statement()
        else:
            right = self.logical()
        res = Assignment(left, op, right)
        return res

    def func_call_statement(self):
        func_name = self.currentToken.value
        self.nextToken()
        if self.currentToken.type != 'LPAREN':
            error.Error("Expected '(' after function call")
            self.hasErrored = True
            exit()
        else:
            self.nextToken()
            actual_params = []
            while self.currentToken != 'EOF' and self.currentToken.type != 'RPAREN':
                actual_params.append(self.param())
                if self.currentToken.type == 'COMMA':
                    self.nextToken()
                    if self.currentToken.type == 'RPAREN':
                        error.Error("Expected a parameter after , (in function call)")
                        self.hasErrored = True
                        exit()
            if self.currentToken == 'EOF':
                error.Error("Expected ')' after function call")
                self.hasErrored = True
                exit()
            if self.currentToken.type == 'RPAREN':
                self.nextToken()
                res = Func_Call_Statement(func_name, actual_params)
            else:
                error.Error("Expected ')' after function call")
                self.hasErrored = True
                exit()
        return res

    def structure(self):
        if self.currentToken.type == 'IF':
            res = self.conditional()
            return res
        elif self.currentToken.type == 'FOR':
            res = self.for_loop()
            return res
        elif self.currentToken.type == 'WHILE':
            res = self.while_loop()
            return res
        elif self.currentToken.type == 'FUNC':
            res = self.function()
            return res
    
    def function(self):
        self.nextToken()
        if self.currentToken.type != 'ID':
            error.Error("Expected a function name after 'func'")
            self.hasErrored = True
            exit()
        else:
            function_name = self.currentToken.value
            self.nextToken()
            if self.currentToken.type != 'LPAREN':
                error.Error("Expected '(' after function name")
                self.hasErrored = True
                exit()
            else:
                self.nextToken()
                formal_params = []
                while self.currentToken != 'EOF' and self.currentToken.type != 'RPAREN' and self.currentToken.type != 'COLON':
                    formal_params.append(self.variable())
                    if self.currentToken.type == 'COMMA':
                        self.nextToken()
                        if self.currentToken.type == 'RPAREN':
                            error.Error("Expected a parameter after , ")
                            self.hasErrored = True
                            exit()
                if self.currentToken == 'EOF':
                    error.Error("Expected ')' after function call")
                    self.hasErrored = True
                    exit()
                if self.currentToken.type == 'RPAREN':
                    self.nextToken()
                    if self.currentToken.type != 'COLON':
                        error.Error("Expected a ':' after function()")
                        self.hasErrored = True
                        exit()
                    else:
                        self.nextToken()
                        funcStatements = []
                        while self.currentToken != 'EOF' and self.currentToken.type != 'END':
                            funcStatements.append(self.statement())
                        if self.currentToken == 'EOF':
                            error.Error("Expected 'END' after a function block")
                            self.hasErrored = True
                            exit()
                        elif self.currentToken.type == 'END':
                            self.nextToken()
                            res = FunctionNode(function_name, formal_params, funcStatements)
                            return res
                        else:
                            error.Error("Expected 'END' after a function block")
                            self.hasErrored = True
                            exit()
                else:
                    error.Error("Expected ')' after function(")
                    self.hasErrored = True
                    exit()

    def for_loop(self):
        self.nextToken()
        if self.currentToken.type != 'LPAREN':
            error.Error("Expected '(' after Repeat")
            self.hasErrored = True
            exit()
        else:
            self.nextToken()
            times_to_loop = self.logical()
            try:
                if times_to_loop.name == 'RPAREN':
                    error.Error("Expected an int inside Repeat( )")
                    self.hasErrored = True 
                    exit()
            except:
                pass
            if self.hasErrored:
                exit()
            if self.currentToken.type != 'RPAREN':
                error.Error("Expected ')' after Repeat(")
                self.hasErrored = True
                exit()
            else:
                self.nextToken()
            if self.currentToken.type != 'COLON':
                error.Error("Expected ':' after Repeat()")
                self.hasErrored = True
                exit()
            else:
                self.nextToken()
                repeatStatements = []
                while self.currentToken != 'EOF' and self.currentToken.type != 'END':
                    repeatStatements.append(self.statement())
                if self.currentToken == 'EOF':
                    error.Error("Expected 'END' after a repeat block")
                    self.hasErrored = True
                    exit()
                elif self.currentToken.type == 'END':
                    self.nextToken()
                    res = ForNode(times_to_loop, repeatStatements)
                    return res
                else:
                    error.Error("Expected 'END' after a repeat block")
                    self.hasErrored = True
                    exit()

    def while_loop(self):
        self.nextToken()
        if self.currentToken.type != 'LPAREN':
            error.Error("Expected '(' after while")
            self.hasErrored = True
            exit()
        else:
            self.nextToken()
            condition = self.logical()
            try:
                if condition.name == 'RPAREN':
                    error.Error("Expected a logical condition inside while( )")
                    self.hasErrored = True 
                    exit()
            except:
                pass
            if self.hasErrored:
                exit()
            if self.currentToken.type != 'RPAREN':
                error.Error("Expected ')' after while(")
                self.hasErrored = True
                exit()
            else:
                self.nextToken()
            if self.currentToken.type != 'COLON':
                error.Error("Expected ':' after while()")
                self.hasErrored = True
                exit()
            else:
                self.nextToken()
                repeatStatements = []
                while self.currentToken != 'EOF' and self.currentToken.type != 'END':
                    repeatStatements.append(self.statement())
                if self.currentToken == 'EOF':
                    error.Error("Expected 'END' after a while block")
                    self.hasErrored = True
                    exit()
                elif self.currentToken.type == 'END':
                    self.nextToken()
                    res = WhileNode(condition, repeatStatements)
                    return res
                else:
                    error.Error("Expected 'END' after a repeat block")
                    self.hasErrored = True
                    exit()

    def conditional(self):
        self.nextToken()
        if self.currentToken.type != 'LPAREN':
            error.Error("Expected '(' after if condition")
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
            error.Error("Expected ':' after if condition")
            self.hasErrored = True
            exit()
        if self.currentToken == 'EOF':
            error.Error("Expected 'END' after a conditional block")
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
                    error.Error("Expected ':' after elif condition")
            if self.currentToken == 'EOF':
                error.Error("Expected 'END' after a conditional block")
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
                        error.Error("Expected ':' after else")
                    if self.currentToken == 'EOF':
                        error.Error("Expected 'END' after a conditional block")
                        self.hasErrored = True
                        exit()
                    elif self.currentToken.type == 'END':
                        self.nextToken()
                        res = ConditionalNode(ifcases, elseStatements)
                        return res
                    else:
                        error.Error("Expected 'END' after a conditional block")
                        self.hasErrored = True
                        exit()
    
    def variable(self):
        res = Variable(self.currentToken)
        self.nextToken()
        return res

    def logical(self):
        leftComp = self.comparison()
        while self.currentToken != 'EOF' and self.currentToken.type in ('AND', 'OR'):
            operator = self.currentToken.type
            self.nextToken()
            rightComp = self.comparison()
            leftComp = LogicalOpNode(leftComp, operator, rightComp)
        return leftComp

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

    def expression(self):
        leftTerm = self.term()
        while self.currentToken != 'EOF' and self.currentToken.type in ('PLUS', 'MINUS'):
            operator = self.currentToken.type
            self.nextToken()
            rightTerm = self.term()
            leftTerm = BinOpNode(leftTerm, operator, rightTerm)
        return leftTerm

    def term(self):
        leftFactor = self.factor()
        while self.currentToken != 'EOF' and self.currentToken.type in ('MUL', 'DIV'):
            operator = self.currentToken.type
            self.nextToken()
            rightFactor = self.factor()
            leftFactor = BinOpNode(leftFactor, operator, rightFactor)
        return leftFactor
    
    #def checkComma(self):
    #    if self.currentToken.type == 'COMMA':
    #        self.nextToken()
    #        temp = self.expression()
     #       return temp
     #   else:
     #       return None

    def factor(self):
        if self.currentToken != 'EOF':
            if (self.currentToken.type == 'PLUS' or self.currentToken.type == 'MINUS'):
                unaryOp = self.currentToken.type
                self.nextToken()
                factor = UnaryOpNode(unaryOp, self.factor())
            elif self.currentToken.type in ('INT', 'FLOAT'):
                factor = NumberNode(self.currentToken.type, self.currentToken.value)
                self.nextToken()
                #com = self.checkComma()
                #if com != None:
                #    return factor, com
            elif self.currentToken.type == 'LPAREN':
                self.nextToken()
                factor = self.logical()
                self.nextToken()
            elif self.currentToken.type == 'STRING':
                factor = StringNode(self.currentToken.value)
                self.nextToken()
                #com = self.checkComma()
                #if com != None:
                 #   return factor, com
            elif self.currentToken.type == 'TRUE':
                factor = NumberNode('INT', 1)
                self.nextToken()
            elif self.currentToken.type == 'FALSE':
                factor = NumberNode('INT', 0)
                self.nextToken()
            elif self.currentToken.type == 'ID':
                factor = self.variable()
            else:
                factor = None
            return factor
        else:
            return None
    
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
                factor = self.variable()
            else:
                factor = None
            return factor
        else:
            return None
