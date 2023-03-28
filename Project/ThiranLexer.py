import globals, error, sys, tamil

PYTHON3 = (sys.version[0] == '3')
if PYTHON3:
    unicode = str

class Token:
    def __init__(self, token_type, token_value=None):
        self.type = token_type
        self.value = token_value
    def __repr__(self):
        return f'{self.type}: {self.value}' if self.value != None else f'{self.type}'


KEYWORDS = {u'காட்டு': Token('PRINT'),
            u'வாங்கு': Token('INPUT'),
            u'மற்றும்': Token('AND'),
            u'அல்லது': Token('OR'),
            u'இல்லை': Token('NOT'),
            u'உண்மை': Token('TRUE'),
            u'பொய்': Token('FALSE'),
            u'if': Token('IF'),
            u'elif': Token('ELIF'),
            u'else': Token('ELSE'),
            u'முடி': Token('END'),
            u'திரும்பச்செய்': Token('FOR'),
            u'உண்மையெனில்': Token('WHILE'),
            u'func': Token('FUNC'),
            u'return': Token('RETURN'),
            u'continue': Token('CONTINUE'),
            u'break': Token('BREAK')}

# ஆம்(true) இல்லை(false)

class Lexer:
    def __init__(self, text):
        self.tokens = []
        self.text = text
        self.pos = -1
        self.currentChar = None
        self.acc_num = ''
        self.acc_str = ''
        self.hasErrored = False

    def nextChar(self):
        self.pos += 1
        if self.pos < len(self.text):
            self.currentChar = self.text[self.pos]
        else:
            self.currentChar = None

    def prevChar(self):
        self.pos -= 1
        if self.pos < len(self.text):
            self.currentChar = self.text[self.pos]
        else:
            self.currentChar = None
    
    def tokenize(self):
        self.nextChar()
        while self.currentChar != None and self.hasErrored == False:
            if self.currentChar in ' \n\t':
                self.handle_number()
            elif tamil.utf8.istamil(self.currentChar) or self.currentChar.isalpha():
                self.handle_id()
                self.prevChar()
            elif self.currentChar in globals.OPERATORS:
                self.handle_number()
                self.handle_operator(self.currentChar)
            elif self.currentChar in globals.DIGITS or self.currentChar == '.':
                self.acc_num += self.currentChar
            elif self.currentChar == ':':
                self.tokens.append(Token('COLON'))
            elif self.currentChar in '()':
                self.handle_number()
                if self.currentChar == '(':
                   self.tokens.append(Token('LPAREN'))
                elif self.currentChar == ')':
                   self.tokens.append(Token('RPAREN'))
            elif self.currentChar == '"':
                self.handle_string()
            elif self.currentChar == '=':
                next_char_eq = self.checkEquals()
                if next_char_eq:
                    self.handle_number()
                    self.tokens.append(Token('EQ'))
                else:
                    self.tokens.append(Token('ASSIGN', '='))
                    self.prevChar()
            elif self.currentChar == '!':
                next_char_eq = self.checkEquals()
                if next_char_eq:
                    self.handle_number()
                    self.tokens.append(Token('NOT_EQ'))
                else:
                    error.Error("Expected '=' after '!'")
                    self.hasErrored = True
            elif self.currentChar == '<':
                next_char_eq = self.checkEquals()
                if next_char_eq:
                    self.handle_number()
                    self.tokens.append(Token('LESS_EQ'))
                else:
                    self.handle_number()
                    self.tokens.append(Token('LESS'))
                    self.prevChar()
            elif self.currentChar == '>':
                next_char_eq = self.checkEquals()
                if next_char_eq:
                    self.handle_number()
                    self.tokens.append(Token('GREATER_EQ'))            
                else:
                    self.handle_number()
                    self.tokens.append(Token('GREATER'))
                    self.prevChar()   
            elif self.currentChar == ',':
                self.handle_number()
                self.tokens.append(Token('COMMA'))
            elif self.currentChar == '#':
                self.nextChar()
                while self.currentChar != '\n' and self.currentChar != None:
                    self.nextChar()
            else:
                error.Error(u"Invalid Character", u""+self.currentChar)
                self.hasErrored =  True
                self.tokens.clear()
            self.nextChar()
        if len(self.acc_num) > 0:
            self.handle_number()
        return self.tokens, self.hasErrored

    def checkEquals(self):
        self.nextChar()
        if self.currentChar == '=':
            #self.nextChar()
            return True
        else:
            return False
    
    def handle_operator(self, op):
        if op == '+':
            self.tokens.append(Token('PLUS'))
        elif op == '-':
            self.tokens.append(Token('MINUS'))
        elif op == '*':
            self.tokens.append(Token('MUL'))
        elif op == '/':
            self.tokens.append(Token('DIV'))
    
    #def handle_paren(self, paren):
    #    if paren == '(':
    #        self.tokens.append(Token('LPAREN'))
    #    elif paren == ')':
    #        self.tokens.append(Token('RPAREN'))
    
    def handle_string(self):
        self.nextChar()
        while self.currentChar != '"':
            if self.currentChar == '\n' or self.currentChar == None:
                error.Error("Unterminated String")
                self.hasErrored = True
                self.tokens.clear()
                return
            if self.currentChar == '\\':
                self.nextChar()
                if self.currentChar == '"':
                    pass
                else:
                    self.prevChar()
            self.acc_str += self.currentChar
            self.nextChar()
        self.tokens.append(Token('STRING', self.acc_str))
        self.acc_str = ''
    
    def handle_number(self):
        if len(self.acc_num) != 0:
            if '.' in self.acc_num:
                if self.acc_num.count('.') > 1 or len(self.acc_num) == 1:
                    error.Error("Invalid Number", self.acc_num)
                    self.hasErrored =  True
                    self.tokens.clear()
                else:
                    self.tokens.append(Token('FLOAT', self.acc_num))
            else:
                self.tokens.append(Token('INT', self.acc_num))
            self.acc_num = ''
    
    def handle_id(self):
        tempID = ""
        while self.currentChar != None and (tamil.utf8.istamil_alnum(self.currentChar) or self.currentChar in tamil.utf8.accent_symbols or self.currentChar in tamil.utf8.pulli_symbols or self.currentChar == '_'):
            if tamil.utf8.istamil(self.currentChar):
                tempID += tamil.utf8.get_letters(self.currentChar)[0]
                self.nextChar()
            elif self.currentChar in tamil.utf8.accent_symbols or self.currentChar in tamil.utf8.pulli_symbols:
                tempID += tamil.utf8.get_letters(self.currentChar)[0]
                self.nextChar()
            else:
                tempID += self.currentChar
                self.nextChar()
        acc_token = KEYWORDS.get(tempID, Token('ID', tempID))
        self.tokens.append(acc_token)

