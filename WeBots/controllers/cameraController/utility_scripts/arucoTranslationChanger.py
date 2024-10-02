import re

# Define your target replacement number
replacement_number = "0.5"

# Regular expression pattern
pattern = r"translation ((-?\d*\.?\d+)|0) ((-?\d*\.?\d+)|0) ((-?\d*\.?\d+)|0)"

# Function to replace non-zero numbers
def replace_non_zero(match):
    # Group 1, 3, and 5 correspond to the three numbers in the translation

    new_values = [
        match.group(1) if match.group(1) == "0" else ("-" + (replacement_number) if int(match.group(1))<0 else replacement_number),
        match.group(3) if match.group(3) == "0" else ("-" + (replacement_number) if int(match.group(3))<0 else replacement_number),
        match.group(5) if match.group(5) == "0" else ("-" + (replacement_number) if int(match.group(5))<0 else replacement_number)
    ]
    print(new_values)
    return f"translation {new_values[0]} {new_values[1]} {new_values[2]}"

# Function to process the entire file
def process_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    with open(output_file, 'w') as file:
        for line in lines:
            # Apply regex substitution only to lines that start with "translation"
            
            if line.lstrip().startswith("translation"):
                new_line = re.sub(pattern, replace_non_zero, line)
                file.write(new_line)
            else:
                # Write the line as is if it doesn't match the "translation" pattern
                file.write(line)

# Example usage:
input_file = 'cameraController/input.txt'  # Path to the input file
output_file = 'cameraController/output.txt'  # Path to the output file
process_file(input_file, output_file)
