# Getting Started with Python


## Programming Language

**A programming language is a set of instructions and syntax used to create software programs.**

Programming languages are the tools we use to write instructions for computers to follow. Computers “think” in binary — strings of 1s and 0s. Programming languages allow us to translate the 1s and 0s into something that humans can understand and write. A programming language is made up of a series of symbols that serves as a bridge that allow humans to translate thoughts into instructions computers can understand.

## Some of the key features of programming languages include:

Syntax: The specific rules and structure used to write code in a programming language.

Data Types: The type of values that can be stored in a program, such as numbers, strings, and booleans.

Variables: Named memory locations that can store values.

Operators: Symbols used to perform operations on values, such as addition, subtraction, and comparison.

Control Structures: Statements used to control the flow of a program, such as if-else statements, loops, and function calls.

Libraries and Frameworks: Collections of pre-written code that can be used to perform common tasks and speed up development.

Examples of popular programming languages include Python, Java, C++, JavaScript, and Ruby. Each language has its own strengths and weaknesses and is suited for different types of projects.


### What is Python?

At its heart, Python is a general purpose programming language that is often regarded as both easy to learn and very user-friendly. While Python is great for scripting and writing code to integrate existing components together, it is also a powerful full-fledged language that is capable of solving pretty much any software problem.

It was created by Guido van Rossum, and released in 1991.

## What is a Python Distribution?
While Python itself is just a programming language, a Python distribution bundles the core language with various other libraries and packages generally geared toward a specific problem domain (such as Data Analytics or Web Development). The standard Python distribution is released on python.org and includes the Python standard library as well as the package manager pip.
Two popular distributions are **Anaconda** and **Miniconda**. 

# Anaconda vs. Miniconda
*Anaconda* is an open source Python distribution that is purpose built for data science, machine learning, and large-scale data processing. It includes the core Python language, over 1,500 data science packages, a package management system called conda, IPython (an interactive Python interpreter), and much more. While it is a very comprehensive distribution, it is also quite large and therefore can take a while to download and consumes a lot of disk space.

**Miniconda** on the other hand, is a slimmed down version of Anaconda and includes all of the same components except for the pre-installed 1,500 data science packages. Instead, we can simply install these packages individually as needed using conda (the Anaconda/Miniconda package manager). We essentially get all the benefits of Anaconda, but without the hassle and burden of the large size. Feel free to use either distribution, however Miniconda is probably the better choice for getting set up with the least amount of fuss.

## Python Libraries

Python libraries are collections of pre-written code and functions that extend the capabilities of the Python programming language. They provide a wide range of tools and modules for various tasks, making it easier for developers to work on specific tasks without reinventing the wheel.

# What is a package manager

A package manager is the tool by which packages can be downloaded, installed, and managed within your projects. It is also responsible for keeping track of package versions and dependencies, packages (Sometimes called libraries) that we don’t install ourselves but are imported by the package we’re installing.
Each Python distribution is usually bundled with a specific package manager. Some of the more common ones are **pip and conda.**


## What can Python do?

Python can be used on a server to create web applications.
Python can be used alongside software to create workflows.
Python can connect to database systems. It can also read and modify files.
Python can be used to handle big data and perform complex mathematics.
Python can be used for rapid prototyping, or for production-ready software development.

## Why Python?

Python works on different platforms (Windows, Mac, Linux, Raspberry Pi, etc).
Python has a simple syntax similar to the English language.
Python has syntax that allows developers to write programs with fewer lines than some other programming languages.
Python runs on an interpreter system, meaning that code can be executed as soon as it is written. This means that prototyping can be very quick.
Python can be treated in a procedural way, an object-oriented way or a functional way.
## What is an IDE?

An integrated development environment (IDE) is a software application that helps programmers develop software code efficiently. It increases developer productivity by combining capabilities such as software editing, building, testing, and packaging in an easy-to-use application. Just as writers use text editors and accountants use spreadsheets, software developers use IDEs to make their job easier.
## Installing Jupyter Notebook

Now that we have a Python distribution installed and were able to run some Python code, let’s install the Jupyter Notebook package. Jupyter Notebook is an open-source web application that allows us to create and share documents that contain live code, equations, visualizations and narrative text. 
# Ways to Install Python and Jupyter Notebook
## Installing Jupyter Notebook on Android

## Installation Steps
Follow these steps to install Jupyter Notebook on your Android device:

Open the Google Play Store on your Android device.

Search for “Pydroid 3 — IDE for Python 3” and install the app.

Once installed, you will need to install Pydroid repository plugin from google play store in order to install libraries in the app

After the installation is complete, open the Pydroid 3 - IDE for Python 3 app

Click on pip icon from the menu on the top left corner

search for jupyter or jupyter-notebook and install it.

Click on the terminal icon for the menu on the top left corner and type the following in the terminal:

jupyter-notebook
This will start the server and open a browser window displaying the Jupyter Notebook interface. You can now create a new notebook and start working on your project
# Windows Miniconda

Go to the Miniconda Download Page. Download the appropriate (32- or 64-Bit) Python 3.X version
of Miniconda.

1. Double click on the .exe file and click Install .
2. Read and agree to the licensing terms.
3. Select if you want to install for ‘Just Me’ or ‘All Users’. If you are installing for ‘All Users’, you
must have Administrator privileges.
4. You will be prompted to select the installation location. By default, Anaconda should try to
install in your home directory. We recommend accepting this default. Click Install .
5. You will be asked if you want to add Anaconda to your PATH environment variable. Do not add
Anaconda to the PATH because it can interfere with other software.
6. You will be asked if you want Anaconda to be your default version of Python. We recommend
‘Yes’. There are some rare instances where you might not make Anaconda the default version,
but they are beyond the scope of this article.
7. Click the Install button.

First, let’s verify that Miniconda was installed. We will use the Anaconda Prompt (miniconda3) Go
to the Start Menu and search for the Anaconda Prompt. Anaconda Prompt is the Command Line
Interface (CLI) we will use with Miniconda ( ). In the shell, type:

```conda --version```

A number should be returned. This is the version number of conda that you have installed.
Now we need to install a few key packages. In the Anaconda Prompt, type

```conda install jupyter```

When prompted with 

‘Proceed ([y]/n)?’ type y

Once the installation completes, in the Anaconda Prompt, type:

```
jupyter notebook
```
A Jupyter Notebook interface will appear in your default browser
An Anaconda Prompt might also open and display a url. If it does, do not close it until you are done
working with Jupyter Notebook. If it does not appear, don’t worry — this has to do with your
operating system and will not affect Jupyter Notebook’s performance.
Let’s install a few packages to get you started working with data in Jupyter Notebook. These are all
Python packages, so we will use pip instead of conda to install them.
We will load:

Matplotlib for visualizations

Seaborn for statistical Visualizations

NLTK for text analysis

Scikit-learn for MAchine Learning


Note that some of these packages depend on other packages. For example, scikit-learn requires
(among others). Package managers (like pip) automatically find all the dependencies and
install them for you. So when you install scikit learn, pip automatically installs NumPy, and any other
dependencies scikit-learn needs.

Open a new Anaconda Prompt and type the following commands (hit ‘Enter’ after each row):

pip install pandas

pip install scikit-learn

pip install matplotlib

pip install seaborn

pip install nltk

pip install beautifulsoup4

You can check to be sure these are all installed by starting Python in your Anaconda Prompt.

On a new line, type python
To check if python is installed write the following command on your device terminal.

python --version
