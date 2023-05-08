# Thiran-Programming-Language
<h1> திறன் நிரலாக்க மொழி </h1>
Thiran (Tamil: திறன் lit."ability") is a simple and easy-to-learn programming language that allows coding in Tamil. The programming language will help tamil-medium school students to develop their problem-solving skills by making code more readable and understandable. It will familiarize them with the concepts and fundamentals of programming before moving on to real-world programming languages. Furthermore, this project will contribute to the growth of the technology industry in Tamil-speaking regions, leading to more diverse and inclusive programming solutions.

## Minimum Requirements for Thiran:
 - Any 64-bit Windows - 7 or higher running computer.
 - 50 MB of minimum disk space for the software.
 - No additional software or program is required to run this standalone software.
 - Note: Thiran is designed to run only on the Windows platform and not on Linux or Mac OS.
## Installing Thiran:
 - Thiran has its own standalone setup which can be downloaded from the Thiran Setup Folder here.
 - Once downloaded, double-click Setup.exe and follow through with the installer, and install Thiran.
 - Finally, to verify if Thiran was installed successfully, open a command prompt terminal anywhere, type the word thiran and press enter. If the terminal shows "திறன் நிரல் மொழி - தொகுத்து  இயக்க ஏதேனும் '.ti' fileயை இணைக்கவும் !", the setup installed correctly.
 - If it doesn't show as above, try adding the installed folder path to the system path environment variable and try again.
## Running your First Program:
 - Now, create a new file with the ".ti" extension anywhere you want (example: myprogram.ti), and type the following code in the file by opening it with any text editor:
 ```
    காட்டு("உலகுக்கு வணக்கம்!")
    காட்டு("தமிழில் ப்ரோக்ராம்மிங் எளிது!")
 ```
 - Save the file and open a command prompt terminal in the same folder where the ".ti" file is, and type the command given below and press enter to execute the file.
 ```
    thiran yourfilename.ti
 ```
 - The terminal outputs the following:
 ```
    உலகுக்கு வணக்கம்!
    தமிழில் ப்ரோக்ராம்மிங் எளிது!
 ```
 - Congratulations, you successfully ran your first Thiran program.

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
> "The fundamental principles that will guide both the education system at large, as well as the individual institutions within it are: [...] promoting multilingualism and the power of language in teaching and learning" - National Education Policy, 2020
