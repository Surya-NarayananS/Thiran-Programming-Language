# Thiran-Programming-Language
# திறன் நிரலாக்க மொழி
Thiran (Tamil: திறன் lit."ability") is a simple and easy-to-learn programming language that allows coding in Tamil. The programming language will help tamil-medium school students to develop their problem-solving skills by making code more readable and understandable. It will familiarize them with the concepts and fundamentals of programming before moving on to real-world programming languages. Furthermore, this project will contribute to the growth of the technology industry in Tamil-speaking regions, leading to more diverse and inclusive programming solutions.

## Example Programs:
### 1. Program to calculate factorial of a number:
![alt text](https://github.com/Surya-NarayananS/Thiran-Programming-Language/blob/efd4813b2562118ebfbe59df3ed529f17ef05fd9/Example%20Program.png)

### 2. Program to calculate area of a rectangle:
```
# செவ்வகத்தின் பரப்பளவை கணக்கிடும் ப்ரோக்ராம்

நீளம் = வாங்கு("செவ்வகத்தின் நீளம் உள்ளிடுக: ")
அகலம் = வாங்கு("செவ்வகத்தின் அகலம் உள்ளிடுக: ")

சரியெனில்(நீளம் < 0 அல்லது அகலம் < 0):
    காட்டு("தவறான மதிப்பு! ")
தவறெனில்:
    பரப்பளவு = நீளம் * அகலம்    # Area = length * breadth
    காட்டு("செவ்வகத்தின் பரப்பளவு:", பரப்பளவு, "அலகுகள்")
முடி
```
### Output:
```
செவ்வகத்தின் நீளம் உள்ளிடுக: 12.5
செவ்வகத்தின் அகலம் உள்ளிடுக: 3
செவ்வகத்தின் பரப்பளவு: 37.5 அலகுகள்
```
-------------------------------------------------------
## Active Todo (currently working):
(As of 27th April, 2023)
- Documentation work.

## Future Todo:
- Exception Handling
- Ability to call functions inside conditionals or inside another function call
- Multi-level indexing
- Multi-line comments
- Bitwise Operators
- standalone math library
- random library
- datetime library
- OS file handling support
- GUI library
- Extend Error class to give more details

## Tasks completed so far:
1. Numbers (int, float)
2. String
3. Unary number (+/-)
4. Errors
5. Arithmetic operators
6. Binary operation evaluation
7. Execute a program file with multiple lines of code
8. Single line comments
9. Operator precedence
10. Comparison operators
11. Logical operators
12. Assignment operators
13. Variables
14. I/O statments
15. Conditionals (if, elif, else)
16. Loops (for, while)
17. Functions
18. Jump statements (return, break, continue)
19. Basic built-in functions (eg: sin(), round(), power(), random(), etc)
20. List Datatype
21. Built-in functions for list manipulation:
    - append_list()
    - remove_from_list()
    - add_to_list()
    - pop_list()
    - list_len()
22. Importing libraries
23. Useful built-in functions for math, random, etc
24. write multiple demo programs
25. Additionals: 
    - Syntax-highlighting & auto-completion support for Thiran (.ti) files in notepad++
    - autohotkey file for automatic conversion of english-typed Thiran keywords into Tamil
26. Documentation works
--------------------------------------------------------------

## Issues Found:
1. [Resolved - 26th march 2023] 
    Outside a loop, when 'break' is used inside 'if' & 'if' evaluates to false
    no error is thrown.
2. [Resolved - 26th march 2023]
    Outside a fn, 'return' works inside loops, conditionals.
3. [Resolved - 8th april 2023]
    'break' doesn't work as expected inside nested loops.
4. [Resolved - 22nd april 2023]
    List manipulation built-in fn calls don't work inside loops (due to pass by reference issue in ThiranInterpreter.py)
#### Update: The language works fine and no issues have been found.
