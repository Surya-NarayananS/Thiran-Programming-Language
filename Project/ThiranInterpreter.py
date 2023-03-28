import ThiranParser, error

def TFEngtoTamil(truthValue):
    if truthValue:
        return u'உண்மை'
    else:
        return u'பொய்'

def TFTamiltoEng(tamilword):
    if tamilword == u'உண்மை':
        return True
    elif tamilword == u'பொய்':
        return False

class Interpreter:
    def __init__(self, ast):
        self.ast = ast
        self.hasErrored = False
        self.SYMBOL_TABLE = {}
        self.LOCAL_SCOPE_SYMBOL_TABLE = {}
        self.DECLARED_FUNCTIONS = {}
        self.insideLoop = False
        self.breakLoop = False
        self.continueLoop = False
        self.insideFunction = False
    
    def interpret(self):
        if self.ast.isEmpty():
            exit()
        return self.statements(self.ast), self.hasErrored

    def binaryOperator(self, node):
        left = node.leftOp
        right = node.rightOp

        leftVal = self.visitNode(left)
        rightVal = self.visitNode(right)
        
        if type(rightVal) == str and type(leftVal) == str:
            if node.op == 'PLUS':
                return leftVal + rightVal
            error.Error("Incompatable Binary Operation with String")
            self.hasErrored = True
            exit()
        elif type(rightVal) == str or type(leftVal) == str:
            error.Error("Incompatable Binary Operation with String")
            self.hasErrored = True
            exit()
        else:
            # Appropriate result for specific operator
            if node.op == 'PLUS':
                return leftVal + rightVal
            elif node.op == 'MINUS':
                return leftVal - rightVal
            elif node.op == 'MUL':
                return leftVal * rightVal
            elif node.op == 'DIV':
                return leftVal / rightVal
    
    def numberValue(self, node):
        type = node.type
        if type == 'INT':
            return int(node.value)
        elif type == 'FLOAT':
            return float(node.value)
    
    def stringValue(self, node):
        return node.value

    def unaryValue(self, node):
        val = self.visitNode(node.exp)
        if node.op == 'PLUS':
            return +val
        elif node.op == 'MINUS':
            return -val
        elif node.op == 'NOT':
            val = (not val)
            return TFEngtoTamil(val)

    def assignStatement(self, node):
        varName = node.left.value
        self.SYMBOL_TABLE[varName] = self.visitNode(node.right)

    # Modified for Function Execution
    def variableValue(self, node):
        varName = node.value
        #val = self.SYMBOL_TABLE.get(varName)
        val = self.LOCAL_SCOPE_SYMBOL_TABLE.get(varName)
        if val == None:
            val = self.SYMBOL_TABLE.get(varName)
            if val == None:
                error.Error('Name Error (Not declared)', varName)
                self.hasErrored = True
                exit()
            else:

                return val
        else:
            return val

    def statements(self, node):
        for statement in node.children:
            self.visitNode(statement)

    def printValue(self, node):
        argument = node.expr
        for i in argument:
            val = self.visitNode(i)
            temp = str(val)
            temp = temp.replace('\\n', '\n')
            temp = temp.replace('\\t', '\t')
            if argument[-1] != i:
                print(temp, end=' ')
            else:
                print(temp)


    def inputValue(self, node):
        display = self.visitNode(node.expr)
        display = str(display)
        display = display.replace('\\n', '\n')
        display = display.replace('\\t', '\t')
        print(display, end='')
        inp = input()
        try:
            inp = eval(inp)
        except:
            print('Detected string input')
        return inp

    def comparisonOperator(self, node):
        left = self.visitNode(node.leftOp)
        right = self.visitNode(node.rightOp)
        operator = node.op
        #'EQ', 'NOT_EQ', 'LESS_EQ', 'LESS', 'GREATER_EQ', 'GREATER'
        if operator == 'EQ':
            val = (left == right)
        elif operator == 'NOT_EQ':
            val = (left != right)
        elif operator == 'LESS_EQ':
            val = (left <= right)
        elif operator == 'LESS':
            val = (left < right)
        elif operator == 'GREATER_EQ':
            val = (left >= right)
        elif operator == 'GREATER':
            val = (left > right)
        return TFEngtoTamil(val)
        
    def logicalOperator(self, node):
        left = self.visitNode(node.leftOp)
        right = self.visitNode(node.rightOp)
        left = TFTamiltoEng(left)
        right = TFTamiltoEng(right)
        operator = node.op
        # AND, OR
        if operator == 'AND':
            val = (left and right)
        elif operator == 'OR':
            val = (left or right)
        return TFEngtoTamil(val)
    
    def conditionalStatement(self, node):
        ifcases = node.ifcases
        elsecase = node.elsecase
        isAnyIfTrue = False
        for i in ifcases:
            for j in i[1]:
                if type(j) == ThiranParser.ReturnNode and not self.insideFunction:
                    error.Error("'return' used outside of function!")
                    self.hasErrored = True
                    exit()
                elif type(j) == ThiranParser.BreakNode and self.insideLoop != True:
                    error.Error("'break' used outside of loop!")
                    self.hasErrored = True
                    exit()
                elif type(j) == ThiranParser.ContinueNode and self.insideLoop != True:
                    error.Error("'continue' used outside of loop!")
                    self.hasErrored = True
                    exit()
        if elsecase != None:
            for i in elsecase:
                if type(i) == ThiranParser.ReturnNode and not self.insideFunction:
                    error.Error("'return' used outside of function!")
                    self.hasErrored = True
                    exit()
                elif type(i) == ThiranParser.BreakNode and self.insideLoop != True:
                    error.Error("'break' used outside of loop!")
                    self.hasErrored = True
                    exit()
                elif type(i) == ThiranParser.ContinueNode and self.insideLoop != True:
                    error.Error("'continue' used outside of loop!")
                    self.hasErrored = True
                    exit()

        for i in ifcases:
            evalCond = self.visitNode(i[0])
            if type(evalCond) == int or type(evalCond) == float:
                if evalCond != 0:
                    evalCond = True
                else:
                    evalCond = False
            else:
                evalCond = TFTamiltoEng(evalCond)
            if type(evalCond) == int:
                if evalCond > 0:
                    evalCond = True
                else:
                    evalCond = False
            if evalCond:
                isAnyIfTrue = True
                for j in i[1]:
                    if type(j) == ThiranParser.ReturnNode:
                        if self.insideFunction:
                            out = self.return_statement(j)
                            if out != None:
                                res = self.visitNode(out)
                                return res
                            else:
                                return 'None'
                        else:
                            error.Error("'return' used outside of function!")
                            self.hasErrored = True
                            exit()
                    elif type(j) == ThiranParser.ConditionalNode or type(j) == ThiranParser.ForNode or type(j) == ThiranParser.WhileNode:
                        res = self.visitNode(j)
                        if res != None:
                            return res
                    elif type(j) == ThiranParser.BreakNode and self.insideLoop == True:
                        self.breakLoop = True
                        return
                    elif type(j) == ThiranParser.BreakNode and self.insideLoop != True:
                        error.Error("'break' used outside of loop!")
                        self.hasErrored = True
                        exit()
                    elif type(j) == ThiranParser.ContinueNode and self.insideLoop == True:
                        self.continueLoop = True
                        return 
                    elif type(j) == ThiranParser.ContinueNode and self.insideLoop != True:
                        error.Error("'continue' used outside of loop!")
                        self.hasErrored = True
                        exit()
                    else:
                        self.visitNode(j) 
                return
        if not isAnyIfTrue:
            if elsecase == None:
                return
            for i in elsecase:
                if type(i) == ThiranParser.ReturnNode:
                    if self.insideFunction:
                        out = self.return_statement(i)
                        if out != None:
                            res = self.visitNode(out)
                            return res
                        else:
                            return 'None'
                    else:
                        error.Error("'return' used outside of function!")
                        self.hasErrored = True
                        exit()
                elif type(i) == ThiranParser.ConditionalNode or type(i) == ThiranParser.ForNode or type(i) == ThiranParser.WhileNode:
                    res = self.visitNode(i)
                    if res != None:
                        return res
                elif type(i) == ThiranParser.BreakNode and self.insideLoop == True:
                    self.breakLoop = True
                    return
                elif type(i) == ThiranParser.BreakNode and self.insideLoop != True:
                    error.Error("'break' used outside of loop!")
                    self.hasErrored = True
                    exit()
                elif type(i) == ThiranParser.ContinueNode and self.insideLoop == True:
                    self.continueLoop = True
                    return 
                elif type(i) == ThiranParser.ContinueNode and self.insideLoop != True:
                    error.Error("'continue' used outside of loop!")
                    self.hasErrored = True
                    exit()
                else:
                    self.visitNode(i)
            return

    def forStatement(self, node):
        times = self.visitNode(node.times_to_loop)
        statements_to_loop = node.repeatStatements
        self.insideLoop = True
        self.breakLoop = False
        self.continueLoop = False
        if type(times) == int and times > 0:
            for i in range(times):
                for j in statements_to_loop:
                    if self.breakLoop:
                        self.insideLoop = False
                        return
                    elif type(j) == ThiranParser.ReturnNode:
                        if self.insideFunction:
                            out = self.return_statement(j)
                            if out != None:
                                res = self.visitNode(out)
                                self.insideLoop = False
                                self.breakLoop = True
                                return res
                            else:
                                self.insideLoop = False
                                self.breakLoop = True
                                return 'None'
                        else:
                            error.Error("'return' used outside of function!")
                            self.hasErrored = True
                            exit()
                    elif type(j) == ThiranParser.ConditionalNode or type(j) == ThiranParser.ForNode or type(j) == ThiranParser.WhileNode:
                        res = self.visitNode(j)
                        if self.continueLoop:
                            self.continueLoop = False
                            break
                        if res != None:
                            self.insideLoop = False
                            self.breakLoop = True
                            return res
                        
                    elif type(j) == ThiranParser.BreakNode:
                        self.breakLoop = True
                        self.insideLoop = False
                        return
                    elif type(j) == ThiranParser.ContinueNode:
                        break
                    else:
                        self.visitNode(j)
        elif type(times) == int and times <= 0:
            self.insideLoop = False
            self.breakLoop = True
            return
        else:
            error.Error("Expected an int type inside repeat( )")
        self.insideLoop = False
        self.breakLoop = True
        return
    
    def whileStatement(self, node):
        repeatStatements = node.repeatStatements
        self.insideLoop = True
        self.breakLoop = False
        self.continueLoop = False
        while self.breakLoop != True:
            condition = self.visitNode(node.condition)
            #print(condition)
            if type(condition) == str:
                condition = TFTamiltoEng(condition)
            #print(condition)
            if condition:
                for i in repeatStatements:
                    if type(i) == ThiranParser.ReturnNode:
                        if self.insideFunction:
                            out = self.return_statement(i)
                            if out != None:
                                res = self.visitNode(out)
                                self.insideLoop = False
                                return res
                            else:
                                self.insideLoop = False
                                return 'None'
                        else:
                            error.Error("'return' used outside of function!")
                            self.hasErrored = True
                            exit()
                    elif type(i) == ThiranParser.ConditionalNode or type(i) == ThiranParser.ForNode or type(i) == ThiranParser.WhileNode:
                        res = self.visitNode(i)
                        if self.continueLoop:
                            self.continueLoop = False
                            break
                        if res != None:
                            self.insideLoop = False
                            return res
                    elif type(i) == ThiranParser.BreakNode:
                        self.insideLoop = False
                        return
                    elif type(i) == ThiranParser.ContinueNode:
                            break
                    else:
                        self.visitNode(i)
            else:
                break
        self.insideLoop = False
        self.breakLoop = True
        return

    def return_statement(self, node):
        returnVal = node.returnVal
        return returnVal

    def func_declaration(self, node):
        func_name = node.function_name
        self.DECLARED_FUNCTIONS[func_name] = node
        return

    def func_call(self, node):
        func_name = node.func_name
        actual_params = node.actual_params
        if func_name in self.DECLARED_FUNCTIONS:
            called_func = self.DECLARED_FUNCTIONS[func_name]
            if len(actual_params) != len(called_func.formal_params):
                err_msg = "You passed " + str(len(actual_params)) + " parameters but function needs " + str(len(called_func.formal_params)) + " parameters"
                error.Error(err_msg)
                self.hasErrored = True
                exit()
            res = self.execute_func(func_name, actual_params)
        else:
            #print('Declared funcs:', self.DECLARED_FUNCTIONS)
            error.Error("Undeclared function called", func_name)
            self.hasErrored = True
            exit()
        return res
    
    def execute_func(self, func_name, actual_params):
        self.insideFunction = True
        called_func = self.DECLARED_FUNCTIONS[func_name]
        if len(called_func.formal_params) != 0:
            for index, param in enumerate(called_func.formal_params):
                self.LOCAL_SCOPE_SYMBOL_TABLE[param.value] = self.visitNode(actual_params[index])
        func_statements = called_func.funcStatements
        #print('Local Scope:', self.LOCAL_SCOPE_SYMBOL_TABLE)
        #print(type(self.LOCAL_SCOPE_SYMBOL_TABLE['num1']))
        for i in func_statements:
            if type(i) == ThiranParser.ReturnNode:
                out = self.return_statement(i)
                if out != None:
                    res = self.visitNode(out)
                    self.insideFunction = False
                    return res
                else:
                    self.insideFunction = False
                    return 'None'
            elif type(i) == ThiranParser.ConditionalNode or type(i) == ThiranParser.ForNode or type(i) == ThiranParser.WhileNode:
                res = self.visitNode(i)
                if res != None:
                    self.insideFunction = False
                    return res
            elif type(i) == ThiranParser.BreakNode:
                error.Error("'break' used outside of loop!")
                self.insideFunction = False
                self.hasErrored = True
                exit()
            elif type(i) == ThiranParser.ContinueNode:
                error.Error("'continue' used outside of loop!")
                self.insideFunction = False
                self.hasErrored = True
                exit()
            else:
                self.visitNode(i)
        self.LOCAL_SCOPE_SYMBOL_TABLE.clear()
        self.insideFunction = False
        return


    def visitNode(self, node):
        node_type = type(node)
        if node_type == ThiranParser.BinOpNode:
            return self.binaryOperator(node)
        if node_type == ThiranParser.ConditionalNode:
            return self.conditionalStatement(node)
        elif node_type == ThiranParser.NumberNode:
            return self.numberValue(node)
        elif node_type == ThiranParser.StringNode:
            return self.stringValue(node)
        elif node_type == ThiranParser.UnaryOpNode:
            return self.unaryValue(node)
        elif node_type == ThiranParser.Assignment:
            return self.assignStatement(node)
        elif node_type == ThiranParser.PrintNode:
            return self.printValue(node)
        elif node_type == ThiranParser.Variable:
            return self.variableValue(node)
        elif node_type == ThiranParser.InputNode:
            return self.inputValue(node)
        elif node_type == ThiranParser.CompOpNode:
            return self.comparisonOperator(node)
        elif node_type == ThiranParser.LogicalOpNode:
            return self.logicalOperator(node)
        elif node_type == ThiranParser.ForNode:
            return self.forStatement(node)
        elif node_type == ThiranParser.WhileNode:
            return self.whileStatement(node)
        elif node_type == ThiranParser.FunctionNode:
            return self.func_declaration(node)
        elif node_type == ThiranParser.Func_Call_Statement:
            return self.func_call(node)
        elif node_type == ThiranParser.ReturnNode:
            return self.return_statement(node)
        else:
            return "Interpreter couldn't handle"
