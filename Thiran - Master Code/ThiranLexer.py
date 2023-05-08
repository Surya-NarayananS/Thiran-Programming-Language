########################################
# Lexer for Thiran Programming Language
########################################

import globals, error, sys, tamil
from sys import exit

PYTHON3 = (sys.version[0] == '3')
if PYTHON3:
    unicode = str

###########################################################
# Token class to instantiate all tokens as tokens objects
###########################################################
class Token:
    def __init__(self, token_type, token_value=None):
        self.type = token_type
        self.value = token_value
    def __repr__(self):
        return f'{self.type}: {self.value}' if self.value != None else f'{self.type}'

#######################################
# Built-in Keywords for THIRAN
#######################################
KEYWORDS = {u'காட்டு': Token('PRINT'),
            u'வாங்கு': Token('INPUT'),
            u'மற்றும்': Token('AND'),
            u'அல்லது': Token('OR'),
            u'இல்லை': Token('NOT'),
            u'சரி': Token('TRUE'),
            u'தவறு': Token('FALSE'),
            u'சரியெனில்': Token('IF'),
            u'அல்லதெனில்': Token('ELIF'),
            u'தவறெனில்': Token('ELSE'),
            u'முடி': Token('END'),
            u'திரும்பச்செய்': Token('FOR'),
            u'உண்மைவரை': Token('WHILE'),
            u'செயல்பாகம்': Token('FUNC'),
            u'வெளிகொடு': Token('RETURN'),
            u'தொடர்': Token('CONTINUE'),
            u'நிறுத்து': Token('BREAK'),
            u'இணை':Token('IMPORT')}

#######################################
# Lexer class to perform Tokenizing
#######################################
class Lexer:
    def __init__(self, text):
        self.tokens = [] # a list to hold all the tokens
        self.text = text
        self.pos = -1
        self.currentChar = None
        self.acc_num = ''
        self.acc_str = ''
        self.hasErrored = False

    # to proceed one character forward
    def nextChar(self): 
        self.pos += 1
        if self.pos < len(self.text):
            self.currentChar = self.text[self.pos]
        else:
            self.currentChar = None

    # to go back to previous character
    def prevChar(self): 
        self.pos -= 1
        if self.pos < len(self.text):
            self.currentChar = self.text[self.pos]
        else:
            self.currentChar = None
    
    # method to perform tokenizing
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
                if(len(str(self.acc_num))) == 1:
                    self.nextChar()
                    if tamil.utf8.istamil(self.currentChar) or self.currentChar.isalpha() or self.currentChar == '_':
                        error.Error("பிழை: தவறான பயன்பாடு", self.acc_num)
                        self.hasErrored =  True
                        exit()
                    self.prevChar()
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
                    error.Error("'!' பின் '=' எதிர்பார்க்கப்பட்டது")
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
            elif self.currentChar == '[':
                self.tokens.append(Token('LSQBR'))
            elif self.currentChar == ']':
                self.handle_number()
                self.tokens.append(Token('RSQBR'))
            else:
                error.Error(u"தவறான எழுத்து", u""+self.currentChar)
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
        elif op == '%':
            self.tokens.append(Token('MOD'))
    
    
    def handle_string(self):
        self.nextChar()
        while self.currentChar != '"':
            if self.currentChar == '\n' or self.currentChar == None:
                error.Error("முடிவற்ற வாக்கியம்")
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
                    error.Error("தவறான எண்", self.acc_num)
                    self.hasErrored =  True
                    self.tokens.clear()
                else:
                    self.tokens.append(Token('FLOAT', self.acc_num))
            else:
                self.tokens.append(Token('INT', self.acc_num))
            self.acc_num = ''
    
    def handle_id(self):
        tempID = ""
        # Checks if the currect character is an alphanumeric character or a tamil character
        # Makes use of tamil library
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

