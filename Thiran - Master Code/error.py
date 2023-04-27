#######################
# Code to print errors
#######################

class Error:
    def __init__(self, description, char=None):
        print('\n' + description + ':', char) if char != None else print(description)