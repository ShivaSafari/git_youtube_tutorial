Sum of Digits: Write a program that calculates the sum of the digits of a number entered by the user.
user_input = input("Enter a number: ")
sum_digits = 0

try:
    number = int(user_input)
    for index, digit_char in enumerate(str(number)):
        digit = int(digit_char)
        sum_digits += digit
    print("Sum of digits:", sum_digits)
except ValueError:
    print("Invalid input. Please enter a valid integer.")

