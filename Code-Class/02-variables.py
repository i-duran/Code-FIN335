#### Variables ####

# Create variables to store any type of data

my_number1 = 1
my_number2 = 2
print("My numbers:", my_number1, "and", my_number2)

# variables cannot contain spaces
# In Python, the convention is to use lowercase letters and underscores to separate words.
# This convention is called "snake_case"

### Basic data types ###

## Strings (str) ##

msg1 = "hello"
msg2 = "world"
print(msg1, msg2)

# # concatenating strings
msg3 = msg1 + " " + msg2 + "!"
print(msg3)
msg3 = f"{msg1} {msg2}!"
print(msg3)

# ## Numbers ##

num1 = 8  # integer (int)
print(type(num1))

num2 = 4.5  # float
print(type(num2))

num3 = 1_056_100.23  # (float)
print(num3)

# print("---Calculations---")

num4 = num1 * num2
print(num4)

# # we can also use this format to put variables inside a string
print(f"The product of {num1} and {num2} is: {num1 * num2}")

print(1 / 3)
print(5 / 2)
print(num1 / num2)

print(f"{num1}/{num2} is: {num1/num2}")

## Raising to the power ##

num = 2**3
print(num)
num = 2 ** (1 / 2)
print(num)

print(2 + 1 * 2)  # guess the result!
