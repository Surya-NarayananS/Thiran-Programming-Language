##############################################
# Interpreter for Thiran Programming Language
##############################################

import ThiranParser, ThiranBuiltIns, ThiranImport, error
from sys import exit

# Function to return tamil words for True & False
def TFEngtoTamil(truthValue):
    if truthValue:
        return u'சரி'
    else:
        return u'தவறு'

# Function to return english words for சரி & தவறு
def TFTamiltoEng(tamilword):
    if tamilword == u'சரி':
        return True
    elif tamilword == u'தவறு':
        return False

############################################################
# Interpreter class to perform interpretation and execution
############################################################

class Interpreter:
    def __init__(self, ast):
        self.ast = ast
        self.hasErrored = False
        
        self.SYMBOL_TABLE = {} # Global symbol table
        self.LOCAL_SCOPE_SYMBOL_TABLE = {}  # Lobal Symbol table (used when inside a function)
        
        # All the built-in functions of Thiran
        # 'func_name' : [no.of.formalparams, Call to function]
        self.BUILT_IN_FUNCTIONS = {'வெளியேறு': [0, ThiranBuiltIns.வெளியேறு],
                                'வகை': [1, ThiranBuiltIns.வகை],
                                'முழுஎண்': [1, ThiranBuiltIns.முழுஎண்],
                                'புள்ளிஎண்': [1, ThiranBuiltIns.புள்ளிஎண்],
                                'வாக்கியம்': [1, ThiranBuiltIns.வாக்கியம்],
                                'round': [1, ThiranBuiltIns.round],
                                'power': [2, ThiranBuiltIns.power],
                                'பட்டியல்_இணை': [2, self.பட்டியல்_இணை],
                                'பட்டியல்_நீக்கு':[2, self.பட்டியல்_நீக்கு],
                                'பட்டியல்_சேர்':[3, self.பட்டியல்_சேர்],
                                'பட்டியல்_சுருக்கு':[1, self.பட்டியல்_சுருக்கு],
                                'பட்டியல்_நீளம்':[1, self.பட்டியல்_நீளம்],
                                'get_pi': [0, ThiranBuiltIns.get_pi],
                                'get_e': [0, ThiranBuiltIns.get_e],
                                'get_tau': [0, ThiranBuiltIns.get_tau],
                                'ceil': [1, ThiranBuiltIns.ceil],
                                'floor': [1, ThiranBuiltIns.floor],
                                'sin': [1, ThiranBuiltIns.sin],
                                'cos': [1, ThiranBuiltIns.cos],
                                'tan': [1, ThiranBuiltIns.tan],
                                'factorial': [1, ThiranBuiltIns.factorial],
                                'mod': [2, ThiranBuiltIns.mod],
                                'gcd': [2, ThiranBuiltIns.gcd],
                                'lcm': [2, ThiranBuiltIns.lcm],
                                'sqrt': [1, ThiranBuiltIns.sqrt],
                                'radians': [1, ThiranBuiltIns.radians],
                                'degrees': [1, ThiranBuiltIns.degrees],
                                'randint': [2, ThiranBuiltIns.randint],
                                'choice': [1, ThiranBuiltIns.choice],
                                'random': [0, ThiranBuiltIns.random]}
        self.DECLARED_FUNCTIONS = {} # Dictionary to hold function definitions defined by the user
        self.insideLoop = False
        self.breakLoop = False
        self.continueLoop = False
        self.insideFunction = False
    
    # method to perform interpretation
    def interpret(self):
        if self.ast.isEmpty():
            exit()
        return self.statements(self.ast), self.hasErrored

    # method to handle binary operation
    def binaryOperator(self, bin_op_node):
        left = bin_op_node.leftOp
        right = bin_op_node.rightOp

        if left == None or right == None:
            error.Error("Binary operationல் பிழை")
            self.hasErrored = True
            exit()

        leftVal = self.visitNode(left)
        rightVal = self.visitNode(right)
        

        if type(rightVal) == str and type(leftVal) == str:
            if bin_op_node.op == 'PLUS':
                return leftVal + rightVal
            error.Error("பிழை: வாக்கியத்துடன் பொருந்தா Binary operation")
            self.hasErrored = True
            exit()
        elif type(rightVal) == str or type(leftVal) == str:
            error.Error("பிழை: வாக்கியத்துடன் பொருந்தா Binary operation")
            self.hasErrored = True
            exit()
        if type(rightVal) == list and type(leftVal) == list:
            if bin_op_node.op == 'PLUS':
                return leftVal + rightVal
            error.Error("பிழை: பட்டியலுடன் பொருந்தா Binary operation")
            self.hasErrored = True
            exit()
        elif type(rightVal) == list or type(leftVal) == list:
            error.Error("பிழை: பட்டியலுடன் பொருந்தா Binary operation")
            self.hasErrored = True
            exit()
        else:
            # Appropriate result for specific operator
            if bin_op_node.op == 'PLUS':
                return leftVal + rightVal
            elif bin_op_node.op == 'MINUS':
                return leftVal - rightVal
            elif bin_op_node.op == 'MUL':
                return leftVal * rightVal
            elif bin_op_node.op == 'DIV':
                if rightVal == 0 or rightVal == 0.0:
                    error.Error("கணித பிழை: வகுக்கும் எண் '0' ஆக இருக்க முடியாது!")
                    self.hasErrored = True
                    exit()
                return leftVal / rightVal
            elif bin_op_node.op == 'MOD':
                return leftVal % rightVal
    
    # method to handle number value
    def numberValue(self, num_node):
        type = num_node.type
        if type == 'INT':
            return int(num_node.value)
        elif type == 'FLOAT':
            return float(num_node.value)
    
    # method to handle string value
    def stringValue(self, str_node):
        return str_node.value
    
    # method to handle unary operation
    def unaryValue(self, unary_node):
        val = self.visitNode(unary_node.exp)
        if unary_node.op == 'PLUS':
            return +val
        elif unary_node.op == 'MINUS':
            return -val
        elif unary_node.op == 'NOT':
            if type(val) != int and type(val)!= float:
                val = TFTamiltoEng(val)
            val = not val
            return TFEngtoTamil(val)
    
    # method to create list datatype
    def listNode(self, list_node):
        temp = list(map(self.visitNode, list_node.elements))
        return temp
    
    # method to handle list indexing
    def listIndexValue(self, list_index_node):
        listelements = self.visitNode(list_index_node.listelement)
        if type(listelements) != list:
            error.Error("பட்டியலில் மட்டுமே indexing செய்ய முடியும்  ")
            self.hasErrored = True
            exit()
        index = self.visitNode(list_index_node.index)
        if type(index) != int:
            error.Error("'பட்டியல்[ ]' உள் ஒரு முழுஎண்(0 அல்லது அதற்கு மேல்) எதிர்பார்க்கப்பட்டது")
            self.hasErrored = True
            exit()
        if index < 0:
            error.Error("'பட்டியல்[ ]' உள் ஒரு முழுஎண்(0 அல்லது அதற்கு மேல்) எதிர்பார்க்கப்பட்டது")
            self.hasErrored = True
            exit()
        try:
            return listelements[index]
        except:
            error.Error("பட்டியல்[ ] index range வெளியே சென்றது", listelements)
            self.hasErrored = True
            exit()
    
    # method to handle assignment statement
    def assignStatement(self, assign_node):
        varName = assign_node.left.value
        if self.insideFunction:
            self.LOCAL_SCOPE_SYMBOL_TABLE[varName] = self.visitNode(assign_node.right)
        else:
            self.SYMBOL_TABLE[varName] = self.visitNode(assign_node.right)
    
    # method to get the value of a variable
    def variableValue(self, var_node):
        varName = var_node.value
        #val = self.SYMBOL_TABLE.get(varName)
        val = self.LOCAL_SCOPE_SYMBOL_TABLE.get(varName)
        if val == None:
            val = self.SYMBOL_TABLE.get(varName)
            if val == None:
                error.Error('அறிவிக்கப்படாத பெயர் பயன்பாடு', varName)
                self.hasErrored = True
                exit()
            else:
                return val
        else:
            return val
    
    # method to execute each statement
    def statements(self, statements_node):
        for statement in statements_node.children:
            self.visitNode(statement)
    
    # method to execute print()
    def printValue(self, print_node):
        argument = print_node.expr
        for i in argument:
            val = self.visitNode(i)
            temp = str(val)
            temp = temp.replace('\\n', '\n')
            temp = temp.replace('\\t', '\t')
            if argument[-1] != i:
                print(temp, end=' ')
            else:
                print(temp)

    # method to evaluate input()
    def inputValue(self, inp_node):
        display = self.visitNode(inp_node.expr)
        display = str(display)
        display = display.replace('\\n', '\n')
        display = display.replace('\\t', '\t')
        print(display, end='')
        inp = input()
        try:
            inp = eval(inp)
        except:
            pass
            #print('Detected string input')
        return inp
    
    # method to handle comparison operation
    def comparisonOperator(self, comp_node):
        left = self.visitNode(comp_node.leftOp)
        right = self.visitNode(comp_node.rightOp)
        operator = comp_node.op
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
    
    # method to handle logical operation
    def logicalOperator(self, logical_node):
        left = self.visitNode(logical_node.leftOp)
        right = self.visitNode(logical_node.rightOp)
        if left == 1.0:
            left = 1
        if right == 1.0:
            right = 1
        if left == 0.0:
            left = 0
        if right == 0.0:
            right = 0
        if type(left) == int and type(right) == int:
            pass
        elif type(left) == int and (type(right) != int and type(right) != float):
            right = TFTamiltoEng(right)
        elif type(right) == int and (type(left) != int and type(left) != float):
            left = TFTamiltoEng(left)
        else:
            left = TFTamiltoEng(left)
            right = TFTamiltoEng(right)
        operator = logical_node.op
        # AND, OR
        if operator == 'AND':
            val = (left and right)
        elif operator == 'OR':
            val = (left or right)
        return TFEngtoTamil(val)
    
    # method to handle import statement
    def importModules(self, imp_node):
        modules_to_import = imp_node.modules_to_import
        for module in modules_to_import:
            [module_symbol_table, module_declared_funcs] = ThiranImport.load_module(module)
            self.SYMBOL_TABLE.update(module_symbol_table)
            self.DECLARED_FUNCTIONS.update(module_declared_funcs)
        return
    
    # method to handle conditional statement (if, elif, else)
    def conditionalStatement(self, cond_node):
        ifcases = cond_node.ifcases
        elsecase = cond_node.elsecase
        isAnyIfTrue = False
        # checking if return break or continue is used outside of a loop or function
        #return - வெளிகொடு
        #break - நிறுத்து 
        #continue - தொடர் 
        for i in ifcases:
            for j in i[1]:
                if type(j) == ThiranParser.ReturnNode and not self.insideFunction:
                    error.Error("செயல்பாகம் வெளியே பயன்பாடு", "வெளிகொடு")
                    self.hasErrored = True
                    exit()
                elif type(j) == ThiranParser.BreakNode and self.insideLoop != True:
                    error.Error("Loopயின் வெளியே பயன்பாடு", "நிறுத்து")
                    self.hasErrored = True
                    exit()
                elif type(j) == ThiranParser.ContinueNode and self.insideLoop != True:
                    error.Error("Loopயின் வெளியே பயன்பாடு", "தொடர்")
                    self.hasErrored = True
                    exit()
        if elsecase != None:
            for i in elsecase:
                if type(i) == ThiranParser.ReturnNode and not self.insideFunction:
                    error.Error("செயல்பாகம் வெளியே பயன்பாடு", "வெளிகொடு")
                    self.hasErrored = True
                    exit()
                elif type(i) == ThiranParser.BreakNode and self.insideLoop != True:
                    error.Error("Loopயின் வெளியே பயன்பாடு", "நிறுத்து")
                    self.hasErrored = True
                    exit()
                elif type(i) == ThiranParser.ContinueNode and self.insideLoop != True:
                    error.Error("Loopயின் வெளியே பயன்பாடு", "தொடர்")
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
                    if self.breakLoop and self.insideLoop:
                        return
                    if type(j) == ThiranParser.ReturnNode:
                        if self.insideFunction:
                            out = self.return_statement(j)
                            if out != None:
                                res = self.visitNode(out)
                                return res
                            else:
                                return 'None'
                        else:
                            error.Error("செயல்பாகம் வெளியே பயன்பாடு", "வெளிகொடு")
                            self.hasErrored = True
                            exit()
                    elif type(j) == ThiranParser.ConditionalNode or type(j) == ThiranParser.ForNode or type(j) == ThiranParser.WhileNode:
                        res = self.visitNode(j)
                        if res != None:
                            return res
                    elif type(j) == ThiranParser.BreakNode and self.insideLoop == True:
                        self.breakLoop = True
                        return 'None' 
                    elif type(j) == ThiranParser.BreakNode and self.insideLoop != True:
                        error.Error("Loopயின் வெளியே பயன்பாடு", "நிறுத்து")
                        self.hasErrored = True
                        exit()
                    elif type(j) == ThiranParser.ContinueNode and self.insideLoop == True:
                        self.continueLoop = True
                        return 'None' 
                    elif type(j) == ThiranParser.ContinueNode and self.insideLoop != True:
                        error.Error("Loopயின் வெளியே பயன்பாடு", "தொடர்")
                        self.hasErrored = True
                        exit()
                    else:
                        self.visitNode(j) 
                return
        if not isAnyIfTrue:
            if elsecase == None:
                return
            for i in elsecase:
                if self.breakLoop and self.insideLoop:
                    return
                if type(i) == ThiranParser.ReturnNode:
                    if self.insideFunction:
                        out = self.return_statement(i)
                        if out != None:
                            res = self.visitNode(out)
                            return res
                        else:
                            return 'None'
                    else:
                        error.Error("செயல்பாகம் வெளியே பயன்பாடு", "வெளிகொடு")
                        self.hasErrored = True
                        exit()
                elif type(i) == ThiranParser.ConditionalNode or type(i) == ThiranParser.ForNode or type(i) == ThiranParser.WhileNode:
                    res = self.visitNode(i)
                    if res != None:
                        return res
                elif type(i) == ThiranParser.BreakNode and self.insideLoop == True:
                    self.breakLoop = True
                    return 'None' 
                elif type(i) == ThiranParser.BreakNode and self.insideLoop != True:
                    error.Error("Loopயின் வெளியே பயன்பாடு", "நிறுத்து")
                    self.hasErrored = True
                    exit()
                elif type(i) == ThiranParser.ContinueNode and self.insideLoop == True:
                    self.continueLoop = True
                    return 'None' 
                elif type(i) == ThiranParser.ContinueNode and self.insideLoop != True:
                    error.Error("Loopயின் வெளியே பயன்பாடு", "தொடர்")
                    self.hasErrored = True
                    exit()
                else:
                    self.visitNode(i)
            return
    
    # method to handle for statement
    def forStatement(self, for_node):
        times = self.visitNode(for_node.times_to_loop)
        statements_to_loop = for_node.repeatStatements
        self.insideLoop = True
        self.breakLoop = False
        self.continueLoop = False
        if type(times) == int and times > 0:
            for i in range(times):
                for j in statements_to_loop:
                    if self.breakLoop:
                        self.insideLoop = False
                        return 'None'
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
                            error.Error("செயல்பாகம் வெளியே பயன்பாடு", "வெளிகொடு")
                            self.hasErrored = True
                            exit()
                    elif type(j) == ThiranParser.ConditionalNode:
                        res = self.visitNode(j)
                        if self.continueLoop:
                            self.continueLoop = False
                            break
                        if res != None:
                            self.insideLoop = False
                            self.breakLoop = True
                            return res
                        if self.breakLoop:
                            self.breakLoop = False
                            self.insideLoop = False
                            return 'None'
                    elif type(j) == ThiranParser.ForNode or type(j) == ThiranParser.WhileNode:#add
                        res = self.visitNode(j)
                        self.breakLoop = False
                        self.insideLoop = True
                        self.continueLoop = False
                        if res != 'None':
                            self.insideLoop = False
                            self.breakLoop = True
                            return res
                    elif type(j) == ThiranParser.BreakNode:
                        self.breakLoop = True
                        self.insideLoop = False
                        return 'None'
                    elif type(j) == ThiranParser.ContinueNode:
                        break
                    else:
                        self.visitNode(j)
        elif type(times) == int and times <= 0:
            self.insideLoop = False
            self.breakLoop = True
            return
        else:
            error.Error("'திரும்பச்செய்()' உள் முழுஎண் எதிர்பார்க்கப்பட்டது")
        self.insideLoop = False
        #self.breakLoop = True
        self.breakLoop = False
        return
    
    # method to handle while statement
    def whileStatement(self, while_node):
        repeatStatements = while_node.repeatStatements
        self.insideLoop = True
        self.breakLoop = False
        self.continueLoop = False
        #while (self.breakLoop != True):
        while (True):
            condition = self.visitNode(while_node.condition)
            if type(condition) == str:
                condition = TFTamiltoEng(condition)
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
                            error.Error("செயல்பாகம் வெளியே பயன்பாடு", "வெளிகொடு")
                            self.hasErrored = True
                            exit()
                    elif type(i) == ThiranParser.ConditionalNode:
                        res = self.visitNode(i)
                        if self.continueLoop:
                            self.continueLoop = False
                            break
                        if self.breakLoop:
                            self.breakLoop = False
                            self.insideLoop = False
                            return 'None'
                        if res != None:
                            self.insideLoop = False
                            return res
                        if self.breakLoop:
                            self.insideLoop = False
                            return 'None'
                    elif type(i) == ThiranParser.ForNode or type(i) == ThiranParser.WhileNode:
                        res = self.visitNode(i)
                        self.breakLoop = False
                        self.insideLoop = True
                        self.continueLoop = False
                        if res != 'None':
                            print(res)
                            self.insideLoop = False
                            return res
                    elif type(i) == ThiranParser.BreakNode:
                        self.breakLoop = True
                        self.insideLoop = False
                        return 'None'
                    elif type(i) == ThiranParser.ContinueNode:
                            break
                    else:
                        self.visitNode(i)
            else:
                break
        self.insideLoop = False
        #self.breakLoop = True
        self.breakLoop = False
        return
    
    # method to handle return statement
    def return_statement(self, ret_node):
        returnVal = ret_node.returnVal
        return returnVal
    
    # method to handle function declaration
    def func_declaration(self, func_dec_node):
        func_name = func_dec_node.function_name
        self.DECLARED_FUNCTIONS[func_name] = func_dec_node
        return
    
    # method to handle a function call
    def func_call(self, func_call_node):
        func_name = func_call_node.func_name
        actual_params = func_call_node.actual_params
        #print('Passed:', actual_params, func_name)
        if func_name in self.DECLARED_FUNCTIONS:
            called_func = self.DECLARED_FUNCTIONS[func_name]
            if len(actual_params) != len(called_func.formal_params):
                err_msg = "தேவை" + str(called_func_params) + "parameters ஆனால் கொடுக்கப்படுள்ளது" + str(len(actual_params)) + "paramenters"
                error.Error(err_msg, str(func_name))
                self.hasErrored = True
                exit()
            res = self.execute_func(func_name, actual_params)
        elif func_name in self.BUILT_IN_FUNCTIONS:
            called_func_params = self.BUILT_IN_FUNCTIONS[func_name][0]
            if len(actual_params) != called_func_params:
                err_msg = "தேவை" + str(called_func_params) + "parameters ஆனால் கொடுக்கப்படுள்ளது" + str(len(actual_params)) + "paramenters"
                error.Error(err_msg, str(func_name))
                self.hasErrored = True
                exit()
            res = self.execute_builtin_func(func_name, actual_params)
        else:
            #print('Declared funcs:', self.DECLARED_FUNCTIONS)
            error.Error("அறிவிக்கப்படாத செயல்பாகம் அழைக்கப்பட்டுள்ளது", func_name)
            self.hasErrored = True
            exit()
        return res
 
    # method to execute a function
    def execute_func(self, func_name, actual_parameters):
        self.insideFunction = True
        called_func = self.DECLARED_FUNCTIONS[func_name]
        if len(called_func.formal_params) != 0:
            for index, param in enumerate(called_func.formal_params):
                self.LOCAL_SCOPE_SYMBOL_TABLE[param.value] = self.visitNode(actual_parameters[index])
        func_statements = called_func.funcStatements
        #print('Local Scope:', self.LOCAL_SCOPE_SYMBOL_TABLE)
        #print(type(self.LOCAL_SCOPE_SYMBOL_TABLE['num1']))
        for i in func_statements:
            if type(i) == ThiranParser.ReturnNode:
                out = self.return_statement(i)
                if out != None:
                    res = self.visitNode(out)
                    self.insideFunction = False
                    self.LOCAL_SCOPE_SYMBOL_TABLE.clear()
                    return res
                else:
                    self.insideFunction = False
                    self.LOCAL_SCOPE_SYMBOL_TABLE.clear()
                    return 'None'
            elif type(i) == ThiranParser.ConditionalNode or type(i) == ThiranParser.ForNode or type(i) == ThiranParser.WhileNode:
                res = self.visitNode(i)
                if res != None:
                    self.insideFunction = False
                    self.LOCAL_SCOPE_SYMBOL_TABLE.clear()
                    return res
            elif type(i) == ThiranParser.BreakNode:
                error.Error("Loopயின் வெளியே பயன்பாடு", "நிறுத்து")
                self.insideFunction = False
                self.hasErrored = True
                exit()
            elif type(i) == ThiranParser.ContinueNode:
                error.Error("Loopயின் வெளியே பயன்பாடு", "தொடர்")
                self.insideFunction = False
                self.hasErrored = True
                exit()
            else:
                self.visitNode(i)
        self.LOCAL_SCOPE_SYMBOL_TABLE.clear()
        self.insideFunction = False
        return
    
    # method to execute builtin functions
    def execute_builtin_func(self, func_name, actual_parameters):
        # check if the builtin func called is for list manipulation
        # பட்டியல்_இணை(), பட்டியல்_நீக்கு(), பட்டியல்_சேர்(), பட்டியல்_சுருக்கு(), பட்டியல்_நீளம்()
        # 'append_list', 'remove_from_list', 'add_to_list', 'pop_list', 'list_len'
        if func_name in ['பட்டியல்_இணை', 'பட்டியல்_நீக்கு', 'பட்டியல்_சேர்', 'பட்டியல்_சுருக்கு', 'பட்டியல்_நீளம்']:
            # Making sure the 1st parameter list variable isn't expanded into list values
            original_list = actual_parameters[0]
            actual_parameters = list(map(self.visitNode, actual_parameters))
            actual_parameters[0] = original_list
        else:
            actual_parameters = list(map(self.visitNode, actual_parameters))
        func = self.BUILT_IN_FUNCTIONS[func_name][1]
        res = func(*actual_parameters)
        return res
    
    # method to execute corresponding methods depending on the type of object
    def visitNode(self, visit_node):
        node_type = type(visit_node)
        if node_type == ThiranParser.BinOpNode:
            return self.binaryOperator(visit_node)
        if node_type == ThiranParser.ConditionalNode:
            return self.conditionalStatement(visit_node)
        elif node_type == ThiranParser.NumberNode:
            return self.numberValue(visit_node)
        elif node_type == ThiranParser.StringNode:
            return self.stringValue(visit_node)
        elif node_type == ThiranParser.UnaryOpNode:
            return self.unaryValue(visit_node)
        elif node_type == ThiranParser.Assignment:
            return self.assignStatement(visit_node)
        elif node_type == ThiranParser.PrintNode:
            return self.printValue(visit_node)
        elif node_type == ThiranParser.Variable:
            return self.variableValue(visit_node)
        elif node_type == ThiranParser.InputNode:
            return self.inputValue(visit_node)
        elif node_type == ThiranParser.CompOpNode:
            return self.comparisonOperator(visit_node)
        elif node_type == ThiranParser.LogicalOpNode:
            return self.logicalOperator(visit_node)
        elif node_type == ThiranParser.ForNode:
            return self.forStatement(visit_node)
        elif node_type == ThiranParser.WhileNode:
            return self.whileStatement(visit_node)
        elif node_type == ThiranParser.FunctionNode:
            return self.func_declaration(visit_node)
        elif node_type == ThiranParser.Func_Call_Statement:
            return self.func_call(visit_node)
        elif node_type == ThiranParser.ReturnNode:
            return self.return_statement(visit_node)
        elif node_type == ThiranParser.ListNode:
            return self.listNode(visit_node)
        elif node_type == ThiranParser.ListIndexNode:
            return self.listIndexValue(visit_node)
        elif node_type == ThiranParser.ImportNode:
            return self.importModules(visit_node)
        else:
            return "தொகுக்க முடியவில்லை! (codeல் பிழை உள்ளதா என சரிபார்க்கவும்)"
    
    ##################################################
    # BUILT-IN LIST FUNCTIONS:
    ##################################################
    
    # பட்டியல்_இணை(), பட்டியல்_நீக்கு(), பட்டியல்_சேர்(), பட்டியல்_சுருக்கு(), பட்டியல்_நீளம்()
    # 'append_list', 'remove_from_list', 'add_to_list', 'pop_list', 'list_len'

    def பட்டியல்_இணை(self, original_list, element_to_add):
        if type(original_list) == ThiranParser.Variable:
            list_name = original_list.value
        original_list = self.visitNode(original_list)
        if type(original_list) != list:
            error.Error("Element ஐ சேர்க்க ஒரு பட்டியல் எதிர்பார்க்கப்பட்டது", "பட்டியல்_இணை()")
            exit()
        else:
            original_list.append(element_to_add)
            if list_name in self.LOCAL_SCOPE_SYMBOL_TABLE:
                self.LOCAL_SCOPE_SYMBOL_TABLE[list_name] = original_list
            elif list_name in self.SYMBOL_TABLE:
                self.SYMBOL_TABLE[list_name] = original_list
            return 'None'

    def பட்டியல்_நீக்கு(self, original_list, element_to_remove):
        if type(original_list) == ThiranParser.Variable:
            list_name = original_list.value
        original_list = self.visitNode(original_list)
        if type(original_list) != list:
            error.Error("Element ஐ நீக்க ஒரு பட்டியல் எதிர்பார்க்கப்பட்டது", "பட்டியல்_நீக்கு()")
            exit()
        else:
            try:
                original_list.remove(element_to_remove)
                if list_name in self.LOCAL_SCOPE_SYMBOL_TABLE:
                    self.LOCAL_SCOPE_SYMBOL_TABLE[list_name] = original_list
                elif list_name in self.SYMBOL_TABLE:
                    self.SYMBOL_TABLE[list_name] = original_list
                return 'None'
            except:
                error.Error("Element பட்டியலில் இல்லை", element_to_remove)
                exit()

    def பட்டியல்_சேர்(self, original_list, index_to_add, element_to_add):
        if type(original_list) == ThiranParser.Variable:
            list_name = original_list.value
        original_list = self.visitNode(original_list)
        if type(original_list) != list:
            error.Error("Element ஐ சேர்க்க ஒரு பட்டியல் எதிர்பார்க்கப்பட்டது", "பட்டியல்_சேர்()")
            exit()
        if type(index_to_add) != int:
            error.Error("முழுஎண் index(0 அல்லது அதற்கு மேல்) எதிர்பார்க்கப்பட்டது", "பட்டியல்_சேர்()")
            exit()
        if index_to_add < 0:
            error.Error("முழுஎண் index(0 அல்லது அதற்கு மேல்) எதிர்பார்க்கப்பட்டது", "பட்டியல்_சேர்()")
            exit()
        try:
            original_list.insert(index_to_add, element_to_add)
            if list_name in self.LOCAL_SCOPE_SYMBOL_TABLE:
                self.LOCAL_SCOPE_SYMBOL_TABLE[list_name] = original_list
            elif list_name in self.SYMBOL_TABLE:
                self.SYMBOL_TABLE[list_name] = original_list
            return 'None'
        except:
            error.Error("தவறான index", "பட்டியல்_சேர்()")
            exit()

    def பட்டியல்_சுருக்கு(self, original_list):
        if type(original_list) == ThiranParser.Variable:
            list_name = original_list.value
        original_list = self.visitNode(original_list)
        if type(original_list) != list:
            error.Error("கடைசி Element ஐ நீக்க ஒரு பட்டியல் எதிர்பார்க்கப்பட்டது", "பட்டியல்_சுருக்கு()")
            exit()
        try:
            pop_ele = original_list.pop()
            if list_name in self.LOCAL_SCOPE_SYMBOL_TABLE:
                self.LOCAL_SCOPE_SYMBOL_TABLE[list_name] = original_list
            elif list_name in self.SYMBOL_TABLE:
                self.SYMBOL_TABLE[list_name] = original_list
            return pop_ele
        except:
            error.Error("கடைசி Element ஐ பட்டியலில் இருந்து நீக்க முடியவில்லை", "பட்டியல்_சுருக்கு()")
            exit()

    def பட்டியல்_நீளம்(self, original_list):
        original_list = self.visitNode(original_list)
        if type(original_list) != list:
            error.Error("நீளம் வெளிகொடுக்க ஒரு பட்டியல் எதிர்பார்க்கப்பட்டது", "பட்டியல்_நீளம்()")
            exit()
        return len(original_list)
